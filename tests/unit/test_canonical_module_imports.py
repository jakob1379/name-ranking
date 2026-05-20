"""Smoke tests for canonical production module paths."""

from scripts import (
    benchmark_classification,
    check_phonetic,
    classify_names,
    final_stats,
    take_screenshots,
)
from scripts import (
    test_classify as test_classify_script,
)

from st_name_ranking import name_normalization, tournament_orchestration
from st_name_ranking.active_learning import lazy_updates, queue, selection
from st_name_ranking.classification import classify_origins, origin_classifier
from st_name_ranking.commands import cli, init_database
from st_name_ranking.interface import app_actions, main, ui
from st_name_ranking.learning import features, model, name_queue
from st_name_ranking.persistence import (
    data_loader,
    database,
    database_io,
    feature_cache,
    feature_store,
    sync_store,
)

MODULE_CONTRACTS = [
    (app_actions, "setup_session_state"),
    (benchmark_classification, "benchmark_classification_batch"),
    (check_phonetic, "main"),
    (classify_names, "main"),
    (classify_origins, "classify_batch"),
    (cli, "app"),
    (data_loader, "load_submodule_json"),
    (database, "init_database"),
    (database_io, "export_database"),
    (feature_cache, "FeatureCache"),
    (feature_store, "get_cached_features"),
    (features, "FeatureExtractor"),
    (final_stats, "main"),
    (init_database, "main"),
    (lazy_updates, "record_comparison_instant"),
    (main, "main"),
    (model, "BradleyTerryModel"),
    (name_normalization, "is_valid_name"),
    (name_queue, "NameQueue"),
    (origin_classifier, "OriginClassifier"),
    (queue, "QueueManager"),
    (selection, "select_candidate_pairs"),
    (sync_store, "sync_names_with_submodule"),
    (take_screenshots, "capture_screenshots"),
    (test_classify_script, "main"),
    (tournament_orchestration, "prepare_tournament_round"),
    (ui, "render_tournament"),
]


def test_canonical_modules_expose_expected_entrypoints():
    missing = [
        f"{module.__name__}.{attribute}" for module, attribute in MODULE_CONTRACTS if not hasattr(module, attribute)
    ]

    assert missing == []
