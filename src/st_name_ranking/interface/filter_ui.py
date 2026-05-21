"""Binary name-filter screen rendering."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import streamlit as st

from st_name_ranking.interface.filter_state import (
    FilterCounts,
    apply_filter_count_transition,
    count_filter_statuses,
    get_excluded_names,
    get_included_names,
    get_undecided_names,
    load_name_inclusions_json,
    set_many_filter_statuses,
)
from st_name_ranking.interface.ui_support import (
    FILTER_SAVE_INTERVAL,
    MAX_EXCLUDED_NAMES_DISPLAY,
    MS_PER_SECOND,
    SLOW_RENDER_THRESHOLD_MS,
    RenderTimer,
)
from st_name_ranking.persistence.database import load_user_setting, save_user_setting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterRenderContext:
    names: list[str]
    inclusions: dict[str, bool]
    undecided_names: list[str]
    progress_placeholder: Any
    stats_placeholder: Any
    name_display_placeholder: Any


def _load_cached_name_inclusions() -> dict[str, bool]:
    cache_key = "name_inclusions_loaded"
    if cache_key in st.session_state:
        return st.session_state.name_inclusions

    try:
        inclusions_json = load_user_setting("name_inclusions", "{}")
        st.session_state.name_inclusions = load_name_inclusions_json(inclusions_json)
        logger.debug("Loaded %d inclusions from database", len(st.session_state.name_inclusions))
    except TypeError:
        st.session_state.name_inclusions = {}
    st.session_state[cache_key] = True
    return st.session_state.name_inclusions


def _persist_name_inclusions(inclusions: dict[str, bool]) -> None:
    _persist_name_inclusions_json(json.dumps(inclusions))


def _persist_name_inclusions_json(inclusions_json: str) -> None:
    save_user_setting("name_inclusions", inclusions_json)


def _current_filter_counts() -> FilterCounts:
    return FilterCounts(
        not_decided=st.session_state.filter_counts_not_decided,
        included=st.session_state.filter_counts_included,
        excluded=st.session_state.filter_counts_excluded,
    )


def _store_filter_counts(counts: FilterCounts, names_hash: str | None = None) -> None:
    st.session_state.filter_counts_not_decided = counts.not_decided
    st.session_state.filter_counts_included = counts.included
    st.session_state.filter_counts_excluded = counts.excluded
    if names_hash is not None:
        st.session_state.filter_counts_names_hash = names_hash


def _update_filter_counts(*, old_status: bool | None, new_status: bool | None) -> None:
    _store_filter_counts(
        apply_filter_count_transition(
            _current_filter_counts(),
            old_status=old_status,
            new_status=new_status,
        ),
    )


def _clear_filter_count_cache() -> None:
    if "filter_counts_names_hash" in st.session_state:
        del st.session_state.filter_counts_names_hash


def _names_filter_hash(names: list[str]) -> str:
    fast_hash = hash((names[0], names[-1], len(names))) if names else hash(0)
    return str(fast_hash)


def _ensure_filter_counts(names: list[str], inclusions: dict[str, bool], names_hash: str) -> None:
    needs_recount = (
        "filter_counts_not_decided" not in st.session_state
        or "filter_counts_included" not in st.session_state
        or "filter_counts_excluded" not in st.session_state
        or st.session_state.get("filter_counts_names_hash") != names_hash
    )
    if not needs_recount:
        return

    logger.debug("Computing filter counts for %d names", len(names))
    count_loop_start = time.perf_counter()
    counts = count_filter_statuses(names, inclusions)
    _store_filter_counts(counts, names_hash)

    logger.debug(
        "Filter counts computed: %d not decided, %d included, %d excluded (%.1fms)",
        counts.not_decided,
        counts.included,
        counts.excluded,
        (time.perf_counter() - count_loop_start) * MS_PER_SECOND,
    )


def _sync_filter_session(names_hash: str) -> None:
    if "filter_names_hash" not in st.session_state or st.session_state.filter_names_hash != names_hash:
        st.session_state.filter_names_hash = names_hash
        st.session_state.filter_index = 0

    if "filter_index" not in st.session_state:
        st.session_state.filter_index = 0


def _current_filter_selection(undecided_names: list[str]) -> tuple[str, int]:
    current_idx = st.session_state.filter_index
    if current_idx >= len(undecided_names):
        current_idx = 0
        st.session_state.filter_index = 0
    return undecided_names[current_idx], current_idx


def _clamp_filter_index(names: list[str]) -> None:
    if st.session_state.filter_index >= len(names):
        st.session_state.filter_index = 0


def _render_filter_name_display(context: FilterRenderContext, current_name: str, current_idx: int) -> None:
    progress = current_idx / len(context.undecided_names)
    context.progress_placeholder.progress(
        progress,
        text=f"Progress: {current_idx + 1} of {len(context.undecided_names)} remaining",
    )

    not_decided = st.session_state.filter_counts_not_decided
    explicitly_included = st.session_state.filter_counts_included
    explicitly_excluded = st.session_state.filter_counts_excluded
    context.stats_placeholder.caption(
        f"Not decided: {not_decided} | Included: {explicitly_included} | Excluded: {explicitly_excluded}",
    )

    if current_name not in context.inclusions:
        border_color = "#757575"
        status_text = "Not decided"
        bg_color = "#FAFAFA"
    elif context.inclusions[current_name]:
        border_color = "#4CAF50"
        status_text = "Included"
        bg_color = "#E8F5E9"
    else:
        border_color = "#F44336"
        status_text = "Excluded"
        bg_color = "#FFEBEE"

    context.name_display_placeholder.markdown(
        f"<div style='border: 4px solid {border_color}; background-color: {bg_color}; "
        f"border-radius: 12px; padding: 20px; text-align: center;'>"
        f"<h1 style='font-size: 72px; margin: 0; color: #212121;'>{current_name}</h1>"
        f"<p style='font-size: 16px; margin: 10px 0 0 0; color: {border_color}; "
        f"font-weight: bold;'>{status_text}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _display_next_undecided_name(context: FilterRenderContext, current_name: str) -> None:
    if current_name in context.undecided_names:
        context.undecided_names.remove(current_name)

    if st.session_state.filter_index >= len(context.undecided_names):
        st.session_state.filter_index = 0

    next_idx = st.session_state.filter_index
    if context.undecided_names:
        _render_filter_name_display(context, context.undecided_names[next_idx], next_idx)
    else:
        st.success("✅ All names processed! Switch to Tournament tab.")


def _apply_filter_decision(
    context: FilterRenderContext,
    current_name: str,
    *,
    status: bool,
    label: str,
    icon: str,
) -> None:
    logger.info("%s %s: %s", icon, label, current_name)
    button_click_start = time.perf_counter()

    old_status = context.inclusions.get(current_name)
    context.inclusions[current_name] = status
    _update_filter_counts(old_status=old_status, new_status=status)
    st.toast(f"{label}: {current_name}", icon=icon)
    st.session_state.last_button_press_time = time.perf_counter()
    _persist_name_inclusions(context.inclusions)
    _display_next_undecided_name(context, current_name)

    logger.debug("%s handled in %.1fms", label, (time.perf_counter() - button_click_start) * MS_PER_SECOND)


def _move_filter_name_to_undecided(context: FilterRenderContext, name: str, source_label: str) -> None:
    old_status = context.inclusions.get(name)
    if old_status is None:
        return

    del context.inclusions[name]
    _update_filter_counts(old_status=old_status, new_status=None)
    logger.info("🔄 %s moved from %s to not decided", name, source_label)
    st.toast(f"{name} moved to not decided", icon="🔄")
    _persist_name_inclusions(context.inclusions)
    st.rerun(scope="fragment")


def _render_filter_decision_buttons(context: FilterRenderContext, current_name: str) -> None:
    col_exclude, col_include = st.columns(2)
    with col_exclude:
        if st.button(
            "Exclude",
            key="exclude_btn",
            width="stretch",
            type="secondary",
            shortcut="Left",
        ):
            _apply_filter_decision(context, current_name, status=False, label="Excluded", icon="👎")
    with col_include:
        if st.button(
            "Include",
            key="include_btn",
            width="stretch",
            type="primary",
            shortcut="Right",
        ):
            _apply_filter_decision(context, current_name, status=True, label="Included", icon="👍")


def _save_filter_periodically(context: FilterRenderContext, current_idx: int) -> None:
    if current_idx % FILTER_SAVE_INTERVAL != 0:
        return

    json_start = time.perf_counter()
    inclusions_json = json.dumps(context.inclusions)
    json_time = (time.perf_counter() - json_start) * MS_PER_SECOND

    save_start = time.perf_counter()
    _persist_name_inclusions_json(inclusions_json)
    save_time = (time.perf_counter() - save_start) * MS_PER_SECOND

    logger.debug("Periodic save: JSON=%.1fms, DB=%.1fms, entries=%d", json_time, save_time, len(context.inclusions))


def _render_filter_batch_buttons(context: FilterRenderContext, current_idx: int) -> None:
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        if st.button("Include All Remaining", type="secondary", help="Include all remaining undecided names"):
            count = set_many_filter_statuses(context.inclusions, context.undecided_names[current_idx:], status=True)
            st.session_state.filter_index = 0  # Reset since undecided list will be empty
            _clear_filter_count_cache()
            st.toast(f"Included {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            _persist_name_inclusions(context.inclusions)
            st.rerun(scope="fragment")
    with col_batch2:
        if st.button("Exclude All Remaining", type="secondary", help="Exclude all remaining undecided names"):
            count = set_many_filter_statuses(context.inclusions, context.undecided_names[current_idx:], status=False)
            st.session_state.filter_index = 0  # Reset since undecided list will be empty
            _clear_filter_count_cache()
            st.toast(f"Excluded {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            _persist_name_inclusions(context.inclusions)
            st.rerun(scope="fragment")


def _render_filter_selection_editor(context: FilterRenderContext) -> None:
    included_names = get_included_names(context.names, context.inclusions)
    excluded_count = st.session_state.filter_counts_excluded

    st.divider()
    st.subheader("📋 Your Selections")

    if included_names:
        sorted_included = sorted(included_names)
        selected_included = st.multiselect(
            f"✅ Included names for tournament ({len(sorted_included)})",
            options=sorted_included,
            default=sorted_included,
            help="Uncheck names to move them back to 'not decided'",
        )

        selected_set = set(selected_included)
        for name in sorted_included:
            if name not in selected_set:
                _move_filter_name_to_undecided(context, name, "included")
    else:
        st.info("No names included yet. Use 'Include' button above to add names.")

    if excluded_count > 0:
        with st.expander(f"❌ Show Excluded Names ({excluded_count})"):
            excluded_names = get_excluded_names(context.names, context.inclusions)
            if excluded_names:
                sorted_excluded = sorted(excluded_names)[:MAX_EXCLUDED_NAMES_DISPLAY]
                remaining = len(excluded_names) - MAX_EXCLUDED_NAMES_DISPLAY

                if remaining > 0:
                    st.caption(f"Showing first {MAX_EXCLUDED_NAMES_DISPLAY} of {len(excluded_names)} excluded names")

                selected_excluded = st.multiselect(
                    "Uncheck names to include them again",
                    options=sorted_excluded,
                    default=sorted_excluded,
                    help="Uncheck names to move them back to 'not decided'",
                )

                selected_set = set(selected_excluded)
                for name in sorted_excluded:
                    if name not in selected_set:
                        _move_filter_name_to_undecided(context, name, "excluded")


@st.fragment
def render_binary_filter(names: list[str]) -> None:
    """Render binary filter interface for including/excluding names.

    Users review names one by one, marking them as included or excluded.
    """
    logger.debug("Filter render started with %d names", len(names))
    timer = RenderTimer.start("Filter")

    if "last_button_press_time" in st.session_state:
        del st.session_state.last_button_press_time

    st.header("Name Filter")
    st.write(
        "Review names one by one. Include names you want to compare in the tournament, "
        "exclude names you don't care about.",
    )
    st.caption(
        "💡 **Keyboard shortcuts**: Left arrow (←) to exclude, Right arrow (→) to include, Space to include",
    )

    progress_placeholder = st.empty()
    stats_placeholder = st.empty()
    name_display_placeholder = st.empty()

    inclusions = _load_cached_name_inclusions()
    timer.log("After inclusions loaded")

    names_hash = _names_filter_hash(names)
    _sync_filter_session(names_hash)
    _clamp_filter_index(names)
    _ensure_filter_counts(names, inclusions, names_hash)
    timer.log("After counts")

    undecided_names = get_undecided_names(names, inclusions)
    if not undecided_names:
        st.success("✅ All names have been processed! Switch to the Tournament tab to compare your selected names.")
        return

    context = FilterRenderContext(
        names=names,
        inclusions=inclusions,
        undecided_names=undecided_names,
        progress_placeholder=progress_placeholder,
        stats_placeholder=stats_placeholder,
        name_display_placeholder=name_display_placeholder,
    )
    current_name, current_idx = _current_filter_selection(undecided_names)

    _render_filter_name_display(context, current_name, current_idx)
    timer.log("After display update")

    _render_filter_decision_buttons(context, current_name)
    _save_filter_periodically(context, current_idx)
    _render_filter_batch_buttons(context, current_idx)
    timer.log("After buttons")

    _render_filter_selection_editor(context)

    timer.log("At end")

    end_time = time.perf_counter()
    elapsed_ms = (end_time - timer.start_time) * MS_PER_SECOND

    if elapsed_ms > SLOW_RENDER_THRESHOLD_MS:
        logger.info("Filter render slow: %.1fms", elapsed_ms)
    else:
        logger.debug("Filter render fast: %.1fms", elapsed_ms)
