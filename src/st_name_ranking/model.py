"""Bradley-Terry model with Bayesian updates for preference learning.

Implements a feature-based Bradley-Terry model with Laplace approximation
for Bayesian updates and Thompson sampling for active learning.
"""

import json
import logging
import pickle
from dataclasses import dataclass

import numpy as np

from st_name_ranking.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Container for model parameters."""

    weight_mean: np.ndarray  # shape (d,)
    weight_cov: np.ndarray  # shape (d, d)
    training_samples: int
    feature_names: list[str]

    def __post_init__(self):
        # Ensure arrays are float64 for stability
        self.weight_mean = self.weight_mean.astype(np.float64)
        self.weight_cov = self.weight_cov.astype(np.float64)

    @property
    def feature_dim(self):
        return self.weight_mean.shape[0]

    def to_blob(self) -> tuple[bytes, bytes, list[str]]:
        """Serialize model state to database BLOB format."""
        weights_blob = pickle.dumps(self.weight_mean)
        cov_blob = pickle.dumps(self.weight_cov)
        return weights_blob, cov_blob, self.feature_names

    @classmethod
    def from_blob(
        cls,
        weights_blob: bytes,
        cov_blob: bytes,
        feature_names: list[str],
        training_samples: int,
    ) -> "ModelState":
        """Deserialize model state from database BLOB format."""
        weight_mean = pickle.loads(weights_blob)
        weight_cov = pickle.loads(cov_blob)
        return cls(weight_mean, weight_cov, training_samples, feature_names)

    @classmethod
    def initialize(
        cls,
        feature_names: list[str],
        prior_variance: float = 1.0,
    ) -> "ModelState":
        """Initialize model with zero mean and isotropic covariance."""
        d = len(feature_names)
        weight_mean = np.zeros(d, dtype=np.float64)
        weight_cov = np.eye(d, dtype=np.float64) * prior_variance
        return cls(weight_mean, weight_cov, 0, feature_names)


class BradleyTerryModel:
    """Feature-based Bradley-Terry model with Bayesian updates.

    Models preference probability as:
        P(i > j) = σ(wᵀ(x_i - x_j))
    where σ is the logistic sigmoid.

    Maintains Gaussian posterior over weights w ~ N(μ, Σ) updated via
    Laplace approximation (iterative reweighted least squares).
    """

    def __init__(self, feature_names: list[str], prior_variance: float = 1.0):
        """Initialize model.

        Args:
            feature_names: Ordered list of feature names
            prior_variance: Prior variance for isotropic Gaussian prior

        """
        self.feature_names = feature_names
        self.d = len(feature_names)
        self.state = ModelState.initialize(feature_names, prior_variance)
        self.prior_variance = prior_variance
        self.rng = np.random.default_rng()

    def sample_utilities(self, features: np.ndarray) -> np.ndarray:
        """Sample utilities for names from posterior predictive distribution.

        Args:
            features: Feature matrix shape (n, d)

        Returns:
            Sampled utilities shape (n,)

        """
        # Sample weights from posterior
        sampled_weights = self.rng.multivariate_normal(
            self.state.weight_mean,
            self.state.weight_cov,
        )
        # Compute utilities
        return features @ sampled_weights

    def update(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        preference: int,
    ):
        """Update model with a single comparison.

        Args:
            features_a: Feature vector for name A
            features_b: Feature vector for name B
            preference: -1 (A preferred), 0 (draw), 1 (B preferred)

        """
        # Convert to batch update
        self.update_batch([(features_a, features_b, preference)])

    def update_batch(
        self,
        comparisons: list[tuple[np.ndarray, np.ndarray, int]],
    ):
        """Update model with a batch of comparisons using IRLS.

        Args:
            comparisons: List of (features_a, features_b, preference)
                where preference: -1 (a preferred), 0 (draw), 1 (b preferred)

        """
        if not comparisons:
            return

        n = len(comparisons)
        d = self.d

        # Design matrix: differences for each comparison
        X = np.zeros((n, d))
        y = np.zeros(n)

        for i, (feat_a, feat_b, pref) in enumerate(comparisons):
            diff = feat_a - feat_b
            X[i] = diff

            # Convert preference to target probability
            if pref == -1:  # A preferred
                y[i] = 1.0
            elif pref == 1:  # B preferred
                y[i] = 0.0
            else:  # draw
                y[i] = 0.5

        # Iterative reweighted least squares (IRLS) for logistic regression
        max_iter = 10
        tol = 1e-6

        w = self.state.weight_mean.copy()
        cov_inv = np.linalg.inv(self.state.weight_cov)

        for iteration in range(max_iter):
            # Current predictions
            eta = X @ w
            p = self._sigmoid(eta)

            # Weights for IRLS
            W = p * (1 - p)
            z = eta + (y - p) / (p * (1 - p) + 1e-10)

            # Solve weighted least squares with prior
            XW = X.T * W
            A = XW @ X + cov_inv
            b = XW @ z + cov_inv @ self.state.weight_mean

            w_new = np.linalg.solve(A, b)

            # Check convergence
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break

            w = w_new

        # Update covariance (inverse of Hessian)
        eta = X @ w
        p = self._sigmoid(eta)
        W = p * (1 - p)
        XW = X.T * W
        posterior_cov_inv = XW @ X + cov_inv
        posterior_cov = np.linalg.inv(posterior_cov_inv)

        # Update state
        self.state.weight_mean = w
        self.state.weight_cov = posterior_cov
        self.state.training_samples += n

    def select_pair(
        self,
        features: np.ndarray,
        names: list[str],
    ) -> tuple[int, int, str, str]:
        """Select pair for active learning using Thompson sampling.

        Strategy: Sample utilities, then select pair where probability
        of preference is closest to 0.5 (maximally uncertain).

        Args:
            features: Feature matrix shape (n, d)
            names: List of names corresponding to rows

        Returns:
            (idx_a, idx_b, name_a, name_b)

        """
        n = len(names)
        if n < 2:
            raise ValueError("Need at least 2 names for pair selection")

        # Sample utilities
        utilities = self.sample_utilities(features)

        # For efficiency, sample a subset of pairs if n is large
        # Simple implementation: random sample of 1000 pairs
        n_candidates = min(1000, n * (n - 1) // 2)
        idx_a = self.rng.choice(n, size=n_candidates, replace=True)
        idx_b = self.rng.choice(n, size=n_candidates, replace=True)

        # Filter out same indices
        mask = idx_a != idx_b
        idx_a = idx_a[mask]
        idx_b = idx_b[mask]

        if len(idx_a) == 0:
            # Fallback: first two names
            return 0, 1, names[0], names[1]

        # Compute acquisition score: uncertainty + diversity bonus
        # Vectorized computation for all candidate pairs
        diff = features[idx_a] - features[idx_b]  # (m, d)
        # mean_score = diff @ self.state.weight_mean  # Not used for score
        var_score = np.einsum(
            "ij,jk,ik->i",
            diff,
            self.state.weight_cov,
            diff,
        )  # (m,)

        utility_diff = np.abs(utilities[idx_a] - utilities[idx_b])  # (m,)
        score = var_score.copy()
        # Penalize trivial comparisons
        score[utility_diff < 0.1] *= 0.5

        best_idx = np.argmax(score)
        i = idx_a[best_idx]
        j = idx_b[best_idx]

        return i, j, names[i], names[j]

    def select_top_k_pairs(
        self,
        features: np.ndarray,
        names: list[str],
        k: int = 3,
    ) -> list[tuple[int, int, str, str]]:
        """Select top K pairs for active learning using Thompson sampling.
        Returns list of (idx_a, idx_b, name_a, name_b) for the top K pairs.
        """
        n = len(names)
        if n < 2:
            raise ValueError("Need at least 2 names for pair selection")
        if k < 1:
            raise ValueError("k must be at least 1")

        # Sample utilities
        utilities = self.sample_utilities(features)

        # Sample candidate pairs
        n_candidates = min(1000, n * (n - 1) // 2)
        total_pairs = n * (n - 1) // 2

        if total_pairs <= n_candidates:
            # Use all possible pairs
            idx_a = np.zeros(total_pairs, dtype=int)
            idx_b = np.zeros(total_pairs, dtype=int)
            pair_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    idx_a[pair_idx] = i
                    idx_b[pair_idx] = j
                    pair_idx += 1
        else:
            # Sample unique random pairs
            pairs_set = set()
            while len(pairs_set) < n_candidates:
                i = self.rng.integers(0, n)
                j = self.rng.integers(0, n)
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                pairs_set.add(pair)
            # Convert to arrays
            pairs_list = list(pairs_set)
            idx_a = np.array([p[0] for p in pairs_list])
            idx_b = np.array([p[1] for p in pairs_list])

        if len(idx_a) == 0:
            # Fallback: first two names repeated
            return [(0, 1, names[0], names[1]) for _ in range(k)]

        # Vectorized computation for all candidate pairs
        diff = features[idx_a] - features[idx_b]  # (m, d)
        var_score = np.einsum(
            "ij,jk,ik->i",
            diff,
            self.state.weight_cov,
            diff,
        )  # (m,)
        utility_diff = np.abs(utilities[idx_a] - utilities[idx_b])  # (m,)
        score = var_score.copy()
        score[utility_diff < 0.1] *= 0.5

        # Get top k indices by score
        if k >= len(score):
            top_indices = np.argsort(score)[::-1]  # all indices
        else:
            top_indices = np.argpartition(score, -k)[-k:]
            # Sort descending
            top_indices = top_indices[np.argsort(score[top_indices])[::-1]]

        # Deduplicate pairs (by normalized index tuple)
        pairs_seen = set()
        result = []
        for idx in top_indices:
            i = idx_a[idx]
            j = idx_b[idx]
            # Normalize pair order
            pair = (min(i, j), max(i, j))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)
            result.append((i, j, names[i], names[j]))
            if len(result) >= k:
                break

        # If we don't have enough pairs, add fallback pairs
        if len(result) < k:
            # Generate all possible unique pairs
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            # Filter out already selected pairs
            available_pairs = [p for p in all_pairs if p not in pairs_seen]

            # If we still don't have enough, we may need to reuse pairs
            # This should rarely happen since n >= 2 and k is small
            needed = k - len(result)
            if len(available_pairs) >= needed:
                # Take the first 'needed' pairs (they're already in deterministic order)
                for i, j in available_pairs[:needed]:
                    result.append((i, j, names[i], names[j]))
            else:
                # Not enough unique pairs - fill with (0, 1) duplicates
                # This only happens when n=2 and k>1
                while len(result) < k:
                    result.append((0, 1, names[0], names[1]))

        return result

    def get_utility(self, features: np.ndarray) -> np.ndarray:
        """Compute expected utility for names.

        Args:
            features: Feature matrix shape (n, d)

        Returns:
            Expected utilities shape (n,)

        """
        return features @ self.state.weight_mean

    def save_to_db(self):
        """Save model state to database."""
        weights_blob, cov_blob, feature_names = self.state.to_blob()

        with get_connection() as conn:
            # Serialize feature names as JSON
            feature_names_json = json.dumps(feature_names)

            conn.execute(
                """
                INSERT OR REPLACE INTO model_state
                (id, feature_weights, uncertainty_matrix, training_samples, feature_names_json, last_updated)
                VALUES (1, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    weights_blob,
                    cov_blob,
                    self.state.training_samples,
                    feature_names_json,
                ),
            )

    def load_from_db(self) -> bool:
        """Load model state from database.

        Returns:
            True if loaded successfully, False if no model exists

        """
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT feature_weights, uncertainty_matrix, training_samples, feature_names_json
                FROM model_state WHERE id = 1
            """)
            row = cursor.fetchone()

            if row is None:
                return False

            weights_blob, cov_blob, training_samples, feature_names_json = row

            if feature_names_json:
                # Load feature names from JSON
                stored_feature_names = json.loads(feature_names_json)
                # Verify dimension matches weights blob
                weight_mean = pickle.loads(weights_blob)
                if len(stored_feature_names) != len(weight_mean):
                    logger.warning(
                        f"Stored feature names count ({len(stored_feature_names)}) "
                        f"does not match weight dimension ({len(weight_mean)}). "
                        f"Model corrupted, reinitializing.",
                    )
                    return False
                # Check if stored feature names match expected feature names
                if set(stored_feature_names) != set(self.feature_names):
                    logger.warning(
                        "Stored feature names differ from expected features. Model outdated, reinitializing.",
                    )
                    return False
                feature_names = stored_feature_names
            else:
                # Fallback to current feature names (for backward compatibility)
                logger.warning(
                    "No feature names stored in database, using current feature names",
                )
                feature_names = self.feature_names

            self.state = ModelState.from_blob(
                weights_blob,
                cov_blob,
                feature_names,
                training_samples,
            )
            # Update self.feature_names to match stored names
            self.feature_names = feature_names
            self.d = len(feature_names)
            return True

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


# Helper functions for database integration


def initialize_model_if_needed(feature_names: list[str]) -> BradleyTerryModel:
    """Initialize or load model from database.

    Returns:
        BradleyTerryModel instance

    """
    model = BradleyTerryModel(feature_names)

    if not model.load_from_db():
        logger.info("No existing model found, initializing new model")
        # Save initial model to database
        model.save_to_db()

    return model
