"""Unit tests for feature extraction API boundaries."""

import pytest

from st_name_ranking.learning.features import FeatureBatchContext, FeatureExtractor


def test_batch_extract_accepts_cache_metadata_only_through_context():
    extractor = FeatureExtractor()

    with pytest.raises(TypeError):
        extractor.batch_extract(["Anna"], name_ids=[1])


def test_batch_extract_validates_context_name_id_length():
    extractor = FeatureExtractor()

    with pytest.raises(ValueError, match="name_ids"):
        extractor.batch_extract(["Anna", "Peter"], context=FeatureBatchContext(name_ids=[1]))
