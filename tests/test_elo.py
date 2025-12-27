"""
Tests for st_name_ranking.elo module.
"""

import pytest

from st_name_ranking import elo


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_equal_ratings(self):
        """Expected score should be 0.5 when ratings are equal."""
        assert elo.expected_score(1500.0, 1500.0) == 0.5
        assert elo.expected_score(1000.0, 1000.0) == 0.5
        assert elo.expected_score(2000.0, 2000.0) == 0.5

    def test_higher_rating_wins(self):
        """Higher rated player should have > 0.5 expected score."""
        # Player A higher rating than B
        e_a = elo.expected_score(1600.0, 1400.0)
        assert e_a > 0.5
        assert e_a < 1.0

        # Verify symmetry
        e_b = elo.expected_score(1400.0, 1600.0)
        assert e_b < 0.5
        assert e_b > 0.0
        assert abs(e_a + e_b - 1.0) < 1e-10

    def test_extreme_ratings(self):
        """Test extreme rating differences."""
        # Very large difference
        e = elo.expected_score(2400.0, 1000.0)
        assert e > 0.99  # Almost certain win
        assert e < 1.0

        # Very small difference
        e = elo.expected_score(1501.0, 1500.0)
        assert e > 0.5
        assert e < 0.51

    def test_expected_score_formula(self):
        """Verify the mathematical formula."""
        # Test specific calculation
        # For R_A = 1500, R_B = 1400: difference = -100
        # exponent = (1400 - 1500) / 400 = -100 / 400 = -0.25
        # 10^(-0.25) ≈ 0.562341
        # E_A = 1 / (1 + 0.562341) ≈ 0.640
        e = elo.expected_score(1500.0, 1400.0)
        expected = 1.0 / (1.0 + 10.0 ** ((1400.0 - 1500.0) / 400.0))
        assert abs(e - expected) < 1e-10


class TestUpdateEloGeneric:
    """Tests for update_elo_generic function."""

    def test_update_win(self):
        """Test updating ratings for a win."""
        ratings = {"Alice": 1500.0, "Bob": 1500.0}
        updated = elo.update_elo_generic(ratings, "Alice", "Bob", 1.0, 0.0)

        # Alice should gain rating, Bob should lose
        assert updated["Alice"] > 1500.0
        assert updated["Bob"] < 1500.0
        # Total rating should be conserved (Elo zero-sum)
        assert abs((updated["Alice"] + updated["Bob"]) - 3000.0) < 1e-10

    def test_update_loss(self):
        """Test updating ratings for a loss."""
        ratings = {"Alice": 1500.0, "Bob": 1500.0}
        updated = elo.update_elo_generic(ratings, "Alice", "Bob", 0.0, 1.0)

        assert updated["Alice"] < 1500.0
        assert updated["Bob"] > 1500.0
        assert abs((updated["Alice"] + updated["Bob"]) - 3000.0) < 1e-10

    def test_update_draw(self):
        """Test updating ratings for a draw."""
        ratings = {"Alice": 1500.0, "Bob": 1500.0}
        updated = elo.update_elo_generic(ratings, "Alice", "Bob", 0.5, 0.5)

        # Draw between equal ratings should not change ratings
        assert abs(updated["Alice"] - 1500.0) < 1e-10
        assert abs(updated["Bob"] - 1500.0) < 1e-10

    def test_update_draw_unequal_ratings(self):
        """Test draw between unequal ratings."""
        ratings = {"Alice": 1600.0, "Bob": 1400.0}
        updated = elo.update_elo_generic(ratings, "Alice", "Bob", 0.5, 0.5)

        # Higher rated player should lose points, lower rated should gain
        assert updated["Alice"] < 1600.0
        assert updated["Bob"] > 1400.0
        # Total rating conserved
        assert abs((updated["Alice"] + updated["Bob"]) - 3000.0) < 1e-10

    def test_custom_k_factor(self):
        """Test using custom K-factor."""
        # Create separate ratings for each call
        ratings1 = {"Alice": 1500.0, "Bob": 1500.0}
        ratings2 = {"Alice": 1500.0, "Bob": 1500.0}
        # Larger K means larger changes
        updated_large_k = elo.update_elo_generic(
            ratings1, "Alice", "Bob", 1.0, 0.0, k=64.0
        )
        updated_small_k = elo.update_elo_generic(
            ratings2, "Alice", "Bob", 1.0, 0.0, k=16.0
        )

        # Larger K should produce larger rating change
        change_large = updated_large_k["Alice"] - 1500.0
        change_small = updated_small_k["Alice"] - 1500.0
        assert change_large > change_small > 0

    def test_missing_player(self):
        """Test that missing players cause no update."""
        ratings = {"Alice": 1500.0}
        # Bob not in ratings
        updated = elo.update_elo_generic(ratings, "Alice", "Bob", 1.0, 0.0)
        # Should return unchanged ratings
        assert updated == ratings

        # Neither player in ratings
        ratings_empty = {}
        updated_empty = elo.update_elo_generic(
            ratings_empty, "Alice", "Bob", 1.0, 0.0
        )
        assert updated_empty == ratings_empty


class TestUpdateElo:
    """Tests for update_elo function (binary win/loss)."""

    def test_win_loss(self):
        """Test standard win/loss update."""
        ratings = {"Alice": 1500.0, "Bob": 1500.0}
        updated = elo.update_elo(ratings, "Alice", "Bob")

        # Alice wins, Bob loses
        assert updated["Alice"] > 1500.0
        assert updated["Bob"] < 1500.0
        assert abs((updated["Alice"] + updated["Bob"]) - 3000.0) < 1e-10

        # Verify it's same as generic with 1.0, 0.0
        generic = elo.update_elo_generic(ratings, "Alice", "Bob", 1.0, 0.0)
        assert updated == generic

    def test_custom_k_factor(self):
        """Test custom K-factor."""
        ratings = {"Alice": 1500.0, "Bob": 1500.0}
        updated = elo.update_elo(ratings, "Alice", "Bob", k=64.0)
        # Should use larger K
        change = updated["Alice"] - 1500.0
        # With K=64, expected change: 64 * (1 - 0.5) = 32
        assert abs(change - 32.0) < 1e-10


class TestUpdateEloDraw:
    """Tests for update_elo_draw function."""

    def test_draw_equal_ratings(self):
        """Draw between equal ratings should not change."""
        ratings = {"Alice": 1500.0, "Bob": 1500.0}
        updated = elo.update_elo_draw(ratings, "Alice", "Bob")

        assert abs(updated["Alice"] - 1500.0) < 1e-10
        assert abs(updated["Bob"] - 1500.0) < 1e-10

    def test_draw_unequal_ratings(self):
        """Draw between unequal ratings should adjust."""
        ratings = {"Alice": 1600.0, "Bob": 1400.0}
        updated = elo.update_elo_draw(ratings, "Alice", "Bob")

        # Higher rated should lose points
        assert updated["Alice"] < 1600.0
        assert updated["Bob"] > 1400.0
        # Total conserved
        assert abs((updated["Alice"] + updated["Bob"]) - 3000.0) < 1e-10

        # Verify it's same as generic with 0.5, 0.5
        generic = elo.update_elo_generic(ratings, "Alice", "Bob", 0.5, 0.5)
        assert updated == generic


class TestInitializeRatings:
    """Tests for initialize_ratings function."""

    def test_initialize_empty(self):
        """Initialize empty list returns empty dict."""
        ratings = elo.initialize_ratings([])
        assert ratings == {}

    def test_initialize_names(self):
        """Initialize with names gives default rating."""
        names = ["Alice", "Bob", "Charlie"]
        ratings = elo.initialize_ratings(names)

        assert len(ratings) == 3
        for name in names:
            assert ratings[name.strip()] == elo.INITIAL_RATING

    def test_initialize_strips_whitespace(self):
        """Names with whitespace should be stripped."""
        names = ["  Alice  ", "Bob\n", "\tCharlie"]
        ratings = elo.initialize_ratings(names)

        assert "Alice" in ratings
        assert "Bob" in ratings
        assert "Charlie" in ratings
        for name in ["Alice", "Bob", "Charlie"]:
            assert ratings[name] == elo.INITIAL_RATING

    def test_initialize_skips_empty_names(self):
        """Empty or whitespace-only names should be skipped."""
        names = ["Alice", "", "  ", "Bob", "\n\n"]
        ratings = elo.initialize_ratings(names)

        assert len(ratings) == 2
        assert "Alice" in ratings
        assert "Bob" in ratings
        assert ratings["Alice"] == elo.INITIAL_RATING
        assert ratings["Bob"] == elo.INITIAL_RATING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
