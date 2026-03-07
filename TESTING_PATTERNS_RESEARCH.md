# Testing Patterns Research Report: Streamlit + SQLite + Bayesian Model

Based on comprehensive research of 2024-2025 best practices, official
documentation (pytest, pytest-playwright, hypothesis), and community consensus.

---

## 1. pytest Fixtures for SQLite Database Testing

### Current State Analysis

Your codebase uses a file-based SQLite database (`data/names.db`). The current
fixture creates temp files but could be improved with in-memory testing for
speed.

### Pattern A: In-Memory Database (Recommended for Unit Tests)

```python
# tests/conftest.py
import sqlite3
import pytest
from pathlib import Path
from contextlib import contextmanager

@pytest.fixture(scope="function")
def in_memory_db():
    """
    Fast, isolated in-memory database for each test.
    Uses shared cache with unique URI for test isolation.
    """
    # Create unique in-memory database per test
    # :memory: with URI allows multiple connections to same DB
    db_uri = "file:memdb1?mode=memory&cache=shared"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    # Initialize schema
    _init_schema(conn)

    yield conn

    conn.close()


def _init_schema(conn):
    """Initialize database schema."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS names (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
            origin_region TEXT,
            phonetic_primary TEXT,
            phonetic_secondary TEXT
        );

        CREATE TABLE IF NOT EXISTS ratings (
            name_id INTEGER PRIMARY KEY REFERENCES names(id),
            rating REAL NOT NULL DEFAULT 1500.0,
            matches INTEGER DEFAULT 0
        );

        -- Add other required tables...
    """)
    conn.commit()
```

### Pattern B: Transaction Rollback for Test Isolation

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

@pytest.fixture(scope="function")
def db_session():
    """
    Database session with automatic transaction rollback.
    Ensures each test starts with clean state.
    """
    # Create engine pointing to test database
    engine = create_engine("sqlite:///:memory:")

    # Create all tables
    Base.metadata.create_all(engine)

    # Connect and begin transaction
    connection = engine.connect()
    transaction = connection.begin()

    # Create session bound to connection
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    # Rollback transaction and close
    session.close()
    transaction.rollback()
    connection.close()
    engine.dispose()


# Alternative: Using nested transactions for SQLite
@pytest.fixture(scope="function")
def sqlite_transaction():
    """
    SQLite-specific transaction rollback using SAVEPOINT.
    More efficient than full rollback for SQLite.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Initialize schema
    _init_schema(conn)

    # Start savepoint
    conn.execute("SAVEPOINT test_savepoint")

    yield conn

    # Rollback to savepoint
    conn.execute("ROLLBACK TO SAVEPOINT test_savepoint")
    conn.execute("RELEASE SAVEPOINT test_savepoint")
    conn.close()
```

### Pattern C: Session-Scoped Database with Function-Scoped Cleanup

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def database_engine():
    """
    Session-scoped database engine (created once per test session).
    Use for expensive setup that's safe to share.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_connection(database_engine):
    """
    Function-scoped connection with transaction rollback.
    Ensures test isolation while sharing schema setup cost.
    """
    connection = database_engine.connect()
    transaction = connection.begin()

    yield connection

    transaction.rollback()
    connection.close()
```

### Common Mistakes to Avoid

1. **Don't use the same :memory: database across threads** - SQLite in-memory is
   not thread-safe across connections
2. **Don't forget to set row_factory** - Without it, you can't access columns by
   name
3. **Don't commit in fixtures** - Let tests control transactions; rollback in
   cleanup
4. **Don't share mutable state** - Each test needs isolated data

### Configuration Best Practices

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers for different test types
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (may use real DB)
    slow: Slow tests to skip in quick mode
    db: Tests that require database

# Database testing configuration
addopts = -v --tb=short
```

---

## 2. pytest-playwright for Streamlit E2E Testing

### Pattern A: Basic Streamlit Test Setup

```python
# tests/test_streamlit_e2e.py
import pytest
from playwright.sync_api import Page, expect

@pytest.fixture(scope="session")
def streamlit_app():
    """
    Start Streamlit app for testing.
    Use subprocess to run 'streamlit run' in background.
    """
    import subprocess
    import time
    import socket

    # Find available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()

    # Start Streamlit
    process = subprocess.Popen(
        ['streamlit', 'run', 'src/st_name_ranking/ui.py', '--server.port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(5)

    url = f"http://localhost:{port}"
    yield url

    # Cleanup
    process.terminate()
    process.wait()


def test_streamlit_loads(page: Page, streamlit_app):
    """Test that Streamlit app loads successfully."""
    page.goto(streamlit_app)

    # Wait for Streamlit to render
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10000)

    # Verify basic elements exist
    expect(page.locator("h1")).to_be_visible()


def test_name_comparison_works(page: Page, streamlit_app):
    """Test the core name comparison functionality."""
    page.goto(streamlit_app)

    # Wait for names to load
    page.wait_for_selector("button:has-text('Name A')", timeout=10000)
    page.wait_for_selector("button:has-text('Name B')", timeout=10000)

    # Click on first name
    page.click("button:has-text('Name A')")

    # Verify next pair loads
    page.wait_for_selector("button:has-text('Name A')", timeout=5000)
```

### Pattern B: Streamlit Testing Framework (Recommended)

Streamlit provides a built-in testing framework that's faster than Playwright
for unit-level UI tests:

```python
# tests/test_streamlit_components.py
from streamlit.testing.v1 import AppTest
import pytest

def test_app_initial_state():
    """Test Streamlit app initial state using testing framework."""
    at = AppTest.from_file("src/st_name_ranking/ui.py")
    at.run()

    # Check no exceptions
    assert not at.exception

    # Check title exists
    assert at.title[0].value == "Name Ranking"


def test_comparison_button_click():
    """Test button interactions."""
    at = AppTest.from_file("src/st_name_ranking/ui.py")
    at.run()

    # Find and click button
    button = at.button[0]
    button.click()
    at.run()

    # Verify state change
    assert not at.exception
```

### Pattern C: Hybrid Approach - Fast Unit + Full E2E

```python
# tests/conftest.py
import pytest

def pytest_addoption(parser):
    """Add option to run full browser tests."""
    parser.addoption(
        "--run-browser",
        action="store_true",
        default=False,
        help="Run full browser-based E2E tests"
    )

@pytest.fixture
def page_context(request):
    """Fixture that provides either mock or real page based on flag."""
    if request.config.getoption("--run-browser"):
        # Return real Playwright page
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            yield page
            browser.close()
    else:
        # Return Streamlit AppTest for fast tests
        from streamlit.testing.v1 import AppTest
        yield AppTest.from_file("src/st_name_ranking/ui.py")
```

### Configuration Best Practices

```python
# conftest.py - Playwright configuration
import pytest

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for all tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "record_video_dir": "test-results/videos/",
    }

@pytest.fixture(scope="function")
def page(context):
    """Create new page for each test with tracing."""
    page = context.new_page()

    # Start tracing for debugging
    page.context.tracing.start(screenshots=True, snapshots=True, sources=True)

    yield page

    # Stop tracing and save
    page.context.tracing.stop(path=f"test-results/trace-{page.url.replace('/', '_')}.zip")
```

---

## 3. Testing Bayesian/Statistical Models (Determinism & Seeds)

### Pattern A: Seeded Random Number Generators

```python
# tests/test_model.py
import numpy as np
import pytest
from st_name_ranking.model import BradleyTerryModel


@pytest.fixture
def seeded_rng():
    """Provide seeded random number generator for reproducibility."""
    return np.random.default_rng(seed=42)


class TestBradleyTerryModel:
    """Tests for Bradley-Terry model with deterministic behavior."""

    def test_model_initialization_deterministic(self):
        """Model initialization should be deterministic."""
        model1 = BradleyTerryModel(["feature1", "feature2"], prior_variance=1.0)
        model2 = BradleyTerryModel(["feature1", "feature2"], prior_variance=1.0)

        # Should have same initial state
        np.testing.assert_array_equal(
            model1.state.weight_mean,
            model2.state.weight_mean
        )

    def test_sampling_with_seed(self, seeded_rng):
        """Sampling with seeded RNG should be reproducible."""
        model = BradleyTerryModel(["f1", "f2"])
        model.rng = seeded_rng

        # Sample twice with same seed
        features = np.array([[1.0, 0.5], [0.5, 1.0]])
        utilities1 = model.sample_utilities(features)

        # Reset RNG with same seed
        model.rng = np.random.default_rng(seed=42)
        utilities2 = model.sample_utilities(features)

        np.testing.assert_array_almost_equal(utilities1, utilities2)

    def test_update_batch_converges(self):
        """Model updates should converge deterministically."""
        model = BradleyTerryModel(["length", "syllables"])

        # Create deterministic training data
        comparisons = [
            (np.array([1.0, 0.5]), np.array([0.5, 0.3]), -1),  # A preferred
            (np.array([0.8, 0.4]), np.array([0.6, 0.6]), 1),   # B preferred
            (np.array([1.2, 0.6]), np.array([0.4, 0.2]), -1),  # A preferred
        ]

        initial_mean = model.state.weight_mean.copy()

        # Update multiple times
        for _ in range(5):
            model.update_batch(comparisons)

        # Weights should have changed
        assert not np.array_equal(model.state.weight_mean, initial_mean)

        # Covariance should remain positive semi-definite
        eigenvalues = np.linalg.eigvals(model.state.weight_cov)
        assert all(eigenvalues >= 0), "Covariance should be positive semi-definite"
```

### Pattern B: Property-Based Testing for Statistical Properties

```python
# tests/test_model_properties.py
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from st_name_ranking.model import BradleyTerryModel


class TestStatisticalProperties:
    """Property-based tests for statistical model invariants."""

    @given(
        st.lists(
            st.tuples(
                st.lists(st.floats(-1.0, 1.0), min_size=2, max_size=5),
                st.lists(st.floats(-1.0, 1.0), min_size=2, max_size=5),
                st.sampled_from([-1, 0, 1])
            ),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_covariance_stays_positive_semidefinite(self, comparisons):
        """
        Property: After any valid updates, covariance matrix must remain
        positive semi-definite (all eigenvalues >= 0).
        """
        # Determine feature dimension from data
        dim = len(comparisons[0][0])
        feature_names = [f"f{i}" for i in range(dim)]

        model = BradleyTerryModel(feature_names)

        # Convert to numpy arrays
        numpy_comparisons = [
            (np.array(c[0]), np.array(c[1]), c[2])
            for c in comparisons
        ]

        # Update model
        model.update_batch(numpy_comparisons)

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(model.state.weight_cov)
        assert all(eigenvalues >= -1e-10), \
            f"Covariance not PSD: min eigenvalue {min(eigenvalues)}"

    @given(
        st.integers(min_value=2, max_value=50),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30)
    def test_pair_selection_returns_valid_indices(self, n_names, k_pairs):
        """
        Property: select_top_k_pairs always returns valid indices
        within the range of available names.
        """
        assume(n_names >= 2)

        dim = 3
        model = BradleyTerryModel(["f1", "f2", "f3"])
        model.rng = np.random.default_rng(seed=42)

        # Generate random features
        features = np.random.randn(n_names, dim)
        names = [f"name_{i}" for i in range(n_names)]

        pairs = model.select_top_k_pairs(features, names, k=k_pairs)

        # Verify all indices are valid
        for i, j, name_a, name_b in pairs:
            assert 0 <= i < n_names, f"Invalid index i={i}"
            assert 0 <= j < n_names, f"Invalid index j={j}"
            assert i != j, "Indices should be different"
            assert name_a == names[i], "Name mismatch"
            assert name_b == names[j], "Name mismatch"
```

### Pattern C: Testing Model Persistence

```python
# tests/test_model_persistence.py
import pytest
import numpy as np
from st_name_ranking.model import BradleyTerryModel


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_save_and_load_preserves_state(self, mock_db):
        """Saving and loading should preserve exact model state."""
        # Create and train model
        original = BradleyTerryModel(["length", "syllables", "vowels"])
        original.state.weight_mean = np.array([0.5, -0.3, 0.8])
        original.state.weight_cov = np.eye(3) * 0.1
        original.state.training_samples = 100

        # Save to DB
        original.save_to_db()

        # Load into new model
        loaded = BradleyTerryModel(["length", "syllables", "vowels"])
        success = loaded.load_from_db()

        assert success
        np.testing.assert_array_equal(
            original.state.weight_mean,
            loaded.state.weight_mean
        )
        np.testing.assert_array_equal(
            original.state.weight_cov,
            loaded.state.weight_cov
        )
        assert original.state.training_samples == loaded.state.training_samples

    def test_load_detects_feature_mismatch(self, mock_db):
        """Loading should fail if feature dimensions don't match."""
        # Save model with 3 features
        model_v1 = BradleyTerryModel(["f1", "f2", "f3"])
        model_v1.save_to_db()

        # Try to load with different features
        model_v2 = BradleyTerryModel(["f1", "f2", "f3", "f4"])
        success = model_v2.load_from_db()

        # Should detect mismatch and not load
        assert not success
```

---

## 4. Property-Based Testing with Hypothesis

### Pattern A: Testing Data Transformations

```python
# tests/test_data_properties.py
from hypothesis import given, strategies as st, assume
from hypothesis.extra.pandas import data_frames, column
import pandas as pd
from st_name_ranking.data_loader import is_valid_name


class TestDataLoaderProperties:
    """Property-based tests for data loading functions."""

    @given(st.text(min_size=1, max_size=50))
    def test_is_valid_name_idempotent(self, name):
        """
        Property: Calling is_valid_name twice should give same result.
        """
        result1 = is_valid_name(name)
        result2 = is_valid_name(name)
        assert result1 == result2

    @given(st.text(alphabet=st.characters(whitelist_categories=('L',)),
                   min_size=1, max_size=30))
    def test_valid_names_only_contain_letters(self, name):
        """
        Property: If is_valid_name returns True, name contains only letters.
        """
        if is_valid_name(name):
            assert name.isalpha()

    @given(st.text(min_size=0, max_size=5))
    def test_short_names_invalid(self, name):
        """
        Property: Names with < 2 characters should be invalid.
        """
        assume(len(name) < 2)
        assert not is_valid_name(name)
```

### Pattern B: Testing Database Operations

```python
# tests/test_database_properties.py
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, invariant
import pytest


class DatabaseStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based test for database operations.
    Tests that database maintains invariants across random operation sequences.
    """

    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        _init_schema(self.conn)
        self.model = {}  # In-memory model for verification

    names = Bundle("names")
    ratings = Bundle("ratings")

    @rule(name=names, target=names)
    def insert_name(self, name):
        """Insert a name into database."""
        try:
            self.conn.execute(
                "INSERT INTO names (name) VALUES (?)",
                (name,)
            )
            self.model[name] = {"rating": 1500.0, "matches": 0}
            return name
        except sqlite3.IntegrityError:
            # Duplicate name - expected behavior
            pass

    @rule(name=names, rating=ratings)
    def update_rating(self, name, rating):
        """Update rating for a name."""
        if name in self.model:
            cursor = self.conn.execute(
                "SELECT id FROM names WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                self.conn.execute(
                    """INSERT OR REPLACE INTO ratings
                       (name_id, rating, matches) VALUES (?, ?, ?)""",
                    (row["id"], rating, self.model[name]["matches"] + 1)
                )
                self.model[name]["rating"] = rating
                self.model[name]["matches"] += 1

    @invariant()
    def ratings_match_model(self):
        """Invariant: Database ratings should match in-memory model."""
        for name, expected in self.model.items():
            cursor = self.conn.execute("""
                SELECT r.rating, r.matches
                FROM names n
                JOIN ratings r ON n.id = r.name_id
                WHERE n.name = ?
            """, (name,))
            row = cursor.fetchone()
            if row:
                assert abs(row["rating"] - expected["rating"]) < 0.001
                assert row["matches"] == expected["matches"]


# Run stateful test
TestDatabaseConsistency = DatabaseStateMachine.TestCase
```

### Pattern C: Composite Strategies for Domain Objects

```python
# tests/strategies.py
from hypothesis import strategies as st
import numpy as np


@st.composite
def comparison_data(draw):
    """Generate valid comparison data for model training."""
    # Random feature dimension between 2 and 10
    dim = draw(st.integers(min_value=2, max_value=10))

    # Generate random feature vectors
    feat_a = np.array(draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=dim,
        max_size=dim
    )))

    feat_b = np.array(draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=dim,
        max_size=dim
    )))

    # Random preference
    preference = draw(st.sampled_from([-1, 0, 1]))

    feature_names = [f"feature_{i}" for i in range(dim)]

    return {
        "feat_a": feat_a,
        "feat_b": feat_b,
        "preference": preference,
        "feature_names": feature_names,
    }


@st.composite
def name_record(draw):
    """Generate valid name records."""
    name = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L',)),
        min_size=2,
        max_size=30
    ))

    gender = draw(st.sampled_from(["Male", "Female", "Unisex"]))

    regions = ["European", "Asian", "African", "American", "Nordic", None]
    origin = draw(st.sampled_from(regions))

    return {
        "name": name.capitalize(),
        "gender": gender,
        "origin_region": origin,
    }


# Usage in tests
@given(comparison_data())
def test_model_accepts_generated_comparisons(data):
    """Model should handle any valid comparison data."""
    from st_name_ranking.model import BradleyTerryModel

    model = BradleyTerryModel(data["feature_names"])
    model.update(data["feat_a"], data["feat_b"], data["preference"])

    # Should not raise any exceptions
    assert model.state.training_samples == 1
```

---

## 5. Test Parametrization Strategies

### Pattern A: Table-Driven Tests with @pytest.mark.parametrize

```python
# tests/test_parametrized.py
import pytest


class TestNameValidation:
    """Parametrized tests for name validation logic."""

    @pytest.mark.parametrize(
        "name,expected_valid",
        [
            ("Anna", True),
            ("Peter", True),
            ("", False),           # Empty string
            ("A", False),          # Too short
            ("Anna123", False),    # Contains numbers
            ("Anna-Marie", True),  # Hyphenated
            ("O'Connor", True),    # Apostrophe
            ("  Anna  ", False),   # Leading/trailing spaces (if not stripped)
        ],
        ids=[
            "valid_simple",
            "valid_male",
            "empty_string",
            "too_short",
            "contains_numbers",
            "hyphenated",
            "apostrophe",
            "with_whitespace",
        ]
    )
    def test_name_validation(self, name, expected_valid):
        """Test various name formats."""
        from st_name_ranking.data_loader import is_valid_name
        result = is_valid_name(name)
        assert result == expected_valid


class TestRatingCalculations:
    """Parametrized tests for Elo rating calculations."""

    @pytest.mark.parametrize(
        "rating_a,rating_b,expected_prob",
        [
            (1500, 1500, 0.5),     # Equal ratings
            (1600, 1400, 0.76),    # A much stronger
            (1400, 1600, 0.24),    # B much stronger
            (2000, 1000, 0.99),    # Extreme difference
        ],
    )
    def test_win_probability(self, rating_a, rating_b, expected_prob):
        """Test Elo win probability calculation."""
        prob = calculate_win_probability(rating_a, rating_b)
        assert abs(prob - expected_prob) < 0.01


class TestDatabaseOperations:
    """Parametrized tests with pytest.param for complex cases."""

    @pytest.mark.parametrize(
        "names,expected_count",
        [
            pytest.param(
                [{"name": "Anna", "gender": "F"}],
                1,
                id="single_insert"
            ),
            pytest.param(
                [{"name": "Anna", "gender": "F"},
                 {"name": "Anna", "gender": "F"}],  # Duplicate
                1,
                id="duplicate_handled"
            ),
            pytest.param(
                [],
                0,
                id="empty_list"
            ),
            pytest.param(
                [{"name": "", "gender": "F"}],  # Invalid name
                0,
                id="invalid_name_filtered",
                marks=pytest.mark.xfail(reason="Empty names should be filtered")
            ),
        ]
    )
    def test_bulk_insert_names(self, db_session, names, expected_count):
        """Test bulk name insertion with various inputs."""
        inserted = insert_names_bulk(db_session, names)
        assert inserted == expected_count
```

### Pattern B: Cross-Product Parametrization

```python
# tests/test_cross_product.py
import pytest


# Generate test matrix: gender x origin
GENDERS = ["Male", "Female", "Unisex"]
ORIGINS = ["European", "Asian", "Nordic", None]

@pytest.mark.parametrize("gender", GENDERS)
@pytest.mark.parametrize("origin", ORIGINS)
def test_filter_combinations(db_session, gender, origin):
    """Test all combinations of gender and origin filters."""
    from st_name_ranking.database import get_names_by_filters

    names = get_names_by_filters(
        gender=gender,
        origins=[origin] if origin else None
    )

    # Should return list (may be empty)
    assert isinstance(names, list)

    # Verify all returned names match filters
    for name in names:
        details = get_name_details(db_session, name)
        if origin:
            assert details["origin_region"] == origin
        if gender != "All":
            assert details["gender"] == gender or details["gender"] == "Unisex"
```

### Pattern C: Fixture Parametrization

```python
# tests/conftest.py
@pytest.fixture(params=["sqlite", "postgresql"])
def database_engine(request):
    """
    Parametrized fixture running tests against multiple database backends.
    """
    if request.param == "sqlite":
        engine = create_engine("sqlite:///:memory:")
    elif request.param == "postgresql":
        engine = create_engine("postgresql://test:test@localhost/test")

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


# Each test using this fixture runs twice - once per database
class TestMultiDatabase:
    def test_insert_and_retrieve(self, database_engine):
        """This test runs against both SQLite and PostgreSQL."""
        # Test logic here...
        pass
```

---

## 6. Mock vs Integration Test Boundaries

### Decision Framework

| Test Type                 | When to Use                                          | When NOT to Use                                               |
| ------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| **Unit (Mock)**           | Pure logic, algorithmic code, data transformations   | Testing SQL query correctness, integration between components |
| **Integration (Real DB)** | Database operations, SQL logic, transaction behavior | Testing unrelated code paths, external APIs                   |
| **E2E (Browser)**         | Critical user flows, cross-component interactions    | Testing error edge cases, exhaustive data scenarios           |

### Pattern A: Mock Boundaries - Test Database Logic

```python
# tests/unit/test_database_logic.py (Use mocks for external deps)
import pytest
from unittest.mock import Mock, patch


class TestDatabaseLogic:
    """Unit tests with mocked external dependencies."""

    def test_update_rating_increments_matches(self):
        """Test rating update logic without real database."""
        # Mock the database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (1,)  # name_id

        with patch('st_name_ranking.database.get_connection',
                   return_value=mock_conn):
            from st_name_ranking.database import update_rating
            update_rating("Anna", 1600.0)

        # Verify the correct SQL was called
        calls = mock_conn.execute.call_args_list
        assert any("INSERT OR REPLACE INTO ratings" in str(c) for c in calls)
```

### Pattern B: Integration - Test Real SQL

```python
# tests/integration/test_database_integration.py (Use real database)
import pytest


class TestDatabaseIntegration:
    """Integration tests with real SQLite database."""

    def test_rating_transaction_atomicity(self, in_memory_db):
        """
        Test that rating updates are atomic.
        MUST use real database to verify transaction behavior.
        """
        # Insert name
        in_memory_db.execute("INSERT INTO names (name) VALUES (?)", ("TestName",))
        name_id = in_memory_db.lastrowid

        # Start transaction
        in_memory_db.execute("BEGIN")
        try:
            in_memory_db.execute(
                "INSERT INTO ratings (name_id, rating) VALUES (?, ?)",
                (name_id, 1600.0)
            )
            # Simulate error
            raise RuntimeError("Simulated error")
        except RuntimeError:
            in_memory_db.rollback()

        # Verify no partial data
        cursor = in_memory_db.execute(
            "SELECT COUNT(*) FROM ratings WHERE name_id = ?",
            (name_id,)
        )
        assert cursor.fetchone()[0] == 0
```

### Pattern C: Test Pyramid for Your Application

```python
# Recommended test distribution for st_name_ranking:

# 70% Unit Tests - Fast, isolated
# tests/unit/
#   - test_model.py (Bradley-Terry logic)
#   - test_features.py (Feature extraction)
#   - test_utils.py (Helper functions)

# 20% Integration Tests - Real DB
# tests/integration/
#   - test_database.py (SQL operations)
#   - test_data_loader.py (File I/O)
#   - test_model_persistence.py (Save/load)

# 10% E2E Tests - Full application
# tests/e2e/
#   - test_streamlit_flows.py (User interactions)
#   - test_end_to_end.py (Complete workflows)
```

### Pattern D: Testing External Dependencies

```python
# tests/conftest.py
@pytest.fixture
def mock_ethnidata():
    """
    Fixture providing mock for EthniData classifier.
    Use for tests that don't need real ML inference.
    """
    with patch("ethnidata.EthniData") as mock:
        instance = mock.return_value
        instance.predict_nationality.return_value = {
            "country_name": "Denmark",
            "confidence": 0.85,
            "country": "DK",
            "region": "Europe",
        }
        yield instance


# Use in unit tests
class TestOriginClassifier:
    def test_classifier_cache(self, mock_ethnidata):
        """Test caching logic without loading real model."""
        from st_name_ranking.origin_classifier import classify_origin

        # First call
        result1 = classify_origin("Anna", use_cache=True)
        # Second call should use cache
        result2 = classify_origin("Anna", use_cache=True)

        # Real classifier only called once
        assert mock_ethnidata.predict_nationality.call_count == 1
```

---

## 7. Test Organization Patterns

### Pattern A: Given-When-Then (BDD Style)

```python
# tests/test_bdd_style.py
class TestNameComparisonWorkflow:
    """
    BDD-style tests using Given-When-Then pattern.
    Makes tests readable as specifications.
    """

    def test_user_selects_preferred_name(self):
        """
        Scenario: User selects preferred name in comparison

        Given two names are displayed for comparison
        And the model has initial ratings for both names
        When the user clicks on the preferred name
        Then the model should update ratings
        And a new pair should be presented
        """
        # Given
        model = BradleyTerryModel(["length", "syllables"])
        name_a, name_b = "Anna", "Peter"
        initial_ratings = {"Anna": 1500.0, "Peter": 1500.0}

        # When
        model.update(
            features_a=np.array([0.5, 0.3]),
            features_b=np.array([0.4, 0.4]),
            preference=-1  # A preferred
        )

        # Then
        assert model.state.training_samples == 1
        assert model.state.weight_mean[0] != 0  # Weights updated

    def test_database_maintains_comparison_history(self):
        """
        Scenario: Comparison history is persisted

        Given a user has made several comparisons
        When I query the comparison history
        Then all comparisons should be returned
        And they should be ordered by timestamp
        """
        # Given
        db = setup_test_database()
        record_comparison(db, "Anna", "Peter", -1)
        record_comparison(db, "Maria", "Lars", 1)
        record_comparison(db, "Anna", "Maria", 0)

        # When
        history = get_comparison_history(db)

        # Then
        assert len(history) == 3
        assert history[0].timestamp <= history[1].timestamp
```

### Pattern B: Arrange-Act-Assert (AAA)

```python
# tests/test_aaa_style.py
class TestRatingCalculations:
    """
    Classic AAA pattern - clear separation of test phases.
    """

    def test_elo_rating_update_after_win(self):
        """Test that winner's rating increases after victory."""
        # Arrange
        winner_rating = 1500.0
        loser_rating = 1500.0
        k_factor = 32

        # Act
        new_winner_rating, new_loser_rating = update_elo_ratings(
            winner_rating, loser_rating, k_factor
        )

        # Assert
        assert new_winner_rating > winner_rating
        assert new_loser_rating < loser_rating
        assert abs((new_winner_rating - winner_rating) -
                   (loser_rating - new_loser_rating)) < 0.01

    def test_feature_extraction_normalizes_values(self):
        """Test that feature values are normalized to [0, 1]."""
        # Arrange
        name = "Alexander"  # 9 letters, 4 syllables

        # Act
        features = extract_features(name)

        # Assert
        assert all(0 <= f <= 1 for f in features.values())
        assert features["length"] == pytest.approx(9 / 20, abs=0.01)
```

### Pattern C: Test Class Organization

```python
# tests/test_model.py
import pytest


class TestBradleyTerryInitialization:
    """Tests for model initialization."""

    def test_initial_weights_are_zero(self): ...
    def test_initial_covariance_is_diagonal(self): ...
    def test_feature_names_stored_correctly(self): ...


class TestBradleyTerryUpdates:
    """Tests for model update behavior."""

    def test_single_comparison_updates_weights(self): ...
    def test_batch_update_more_efficient(self): ...
    def test_draw_updates_uncertainty(self): ...


class TestBradleyTerrySampling:
    """Tests for Thompson sampling."""

    def test_sampling_with_zero_uncertainty_deterministic(self): ...
    def test_pair_selection_prioritizes_uncertain_comparisons(self): ...
    def test_cross_cluster_pairs_improve_diversity(self): ...


class TestBradleyTerryPersistence:
    """Tests for save/load functionality."""

    def test_save_preserves_all_state(self): ...
    def test_load_detects_version_mismatch(self): ...
    def test_migration_handles_old_format(self): ...
```

---

## Summary: Recommended Testing Architecture

### File Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/                       # Unit tests (70%)
│   ├── test_model.py          # Bradley-Terry model
│   ├── test_features.py       # Feature extraction
│   ├── test_utils.py          # Utility functions
│   └── test_data_loader.py    # Data loading logic
├── integration/               # Integration tests (20%)
│   ├── test_database.py       # SQLite operations
│   ├── test_model_db.py       # Model persistence
│   └── test_data_pipeline.py  # End-to-end data flow
├── e2e/                       # E2E tests (10%)
│   ├── test_streamlit.py      # Streamlit UI
│   └── test_workflows.py      # Complete user flows
└── properties/                # Property-based tests
    ├── test_model_properties.py
    └── test_data_properties.py
```

### Key Libraries to Add

```txt
# requirements-test.txt
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-playwright>=0.4.0
hypothesis>=6.100.0
pytest-cov>=4.1.0
pytest-xdist>=3.5.0
freezegun>=1.4.0
responses>=0.25.0
```

### Running Tests

```bash
# Run only fast unit tests
pytest tests/unit -xvs

# Run integration tests with real database
pytest tests/integration -xvs --run-integration

# Run E2E tests
pytest tests/e2e --run-playwright

# Run property-based tests with more examples
pytest tests/properties --hypothesis-profile=ci

# Run with coverage
pytest --cov=st_name_ranking --cov-report=html

# Parallel execution
pytest -n auto
```

### CI Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run unit tests
        run: pytest tests/unit -xvs --cov --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -xvs

      - name: Run property tests
        run: pytest tests/properties --hypothesis-profile=ci

      # E2E tests only on main branch
      - name: Run E2E tests
        if: github.ref == 'refs/heads/main'
        run: pytest tests/e2e --run-playwright
```

This architecture provides comprehensive coverage while maintaining fast
feedback loops during development.
