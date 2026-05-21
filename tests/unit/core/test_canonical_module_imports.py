"""Smoke tests for canonical production module paths."""

# Direct module imports are intentional: this test guards every canonical path,
# and the scanner only credits explicit module import edges.
# ruff: noqa: PLR0402

import scripts.benchmark_classification as benchmark_classification
import scripts.check_phonetic as check_phonetic
import scripts.classify_names as classify_names
import scripts.final_stats as final_stats
import scripts.take_screenshots as take_screenshots
import scripts.test_classify as test_classify_script

import st_name_ranking.active_learning.lazy_updates as lazy_updates
import st_name_ranking.active_learning.queue as queue
import st_name_ranking.active_learning.selection as selection
import st_name_ranking.classification.classify_origins as classify_origins
import st_name_ranking.classification.origin_classifier as origin_classifier
import st_name_ranking.commands.cli as cli
import st_name_ranking.commands.init_database as init_database
import st_name_ranking.interface.app_actions as app_actions
import st_name_ranking.interface.filter_ui as filter_ui
import st_name_ranking.interface.main as main
import st_name_ranking.interface.name_queue as name_queue
import st_name_ranking.interface.rankings_ui as rankings_ui
import st_name_ranking.interface.similarity as similarity
import st_name_ranking.interface.similarity_ui as similarity_ui
import st_name_ranking.interface.tournament_orchestration as tournament_orchestration
import st_name_ranking.interface.tournament_ui as tournament_ui
import st_name_ranking.interface.ui as ui
import st_name_ranking.learning.features as features
import st_name_ranking.learning.model as model
import st_name_ranking.persistence.data_loader as data_loader
import st_name_ranking.persistence.database as database
import st_name_ranking.persistence.database_io as database_io
import st_name_ranking.persistence.feature_cache as feature_cache
import st_name_ranking.persistence.feature_store as feature_store
import st_name_ranking.persistence.name_normalization as name_normalization
import st_name_ranking.persistence.sync_store as sync_store

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
    (filter_ui, "render_binary_filter"),
    (final_stats, "main"),
    (init_database, "main"),
    (lazy_updates, "record_comparison_instant"),
    (main, "main"),
    (model, "BradleyTerryModel"),
    (name_normalization, "is_valid_name"),
    (name_queue, "NameQueue"),
    (origin_classifier, "OriginClassifier"),
    (queue, "QueueManager"),
    (rankings_ui, "render_rankings"),
    (selection, "select_candidate_pairs"),
    (similarity, "get_string_similarity_scores"),
    (similarity_ui, "render_similarity"),
    (sync_store, "sync_names_with_submodule"),
    (take_screenshots, "capture_screenshots"),
    (test_classify_script, "main"),
    (tournament_orchestration, "prepare_tournament_round"),
    (tournament_ui, "render_tournament"),
    (ui, "render_tournament"),
]


def test_canonical_modules_expose_expected_entrypoints():
    missing = [
        f"{module.__name__}.{attribute}" for module, attribute in MODULE_CONTRACTS if not hasattr(module, attribute)
    ]

    assert missing == []
