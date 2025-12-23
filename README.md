# Name Ranking Application

A Streamlit-based web application for ranking Danish names using Elo rating system, with similarity search and origin classification.

## Features

### 🏆 Elo Rating Tournament
- Compare two random names side-by-side
- Vote for preferred name or mark as draw
- Real-time Elo rating updates
- Top 10 rankings display
- Keyboard shortcuts (← → ↑ arrows)

### 🔍 Similarity Search
- **String similarity** (Levenshtein distance) for name matching
- **Vector similarity** (LLM embeddings) for semantic matching
- Find names similar to a reference name

### 🌍 Origin Classification
- Automatic nationality prediction using `name2nat`
- Mapping to geographic regions (Nordic, European, Asian, etc.)
- Confidence scoring for predictions
- Batch processing for unclassified names

### ⚙️ Filtering & Management
- Gender filtering (Male, Female, Unisex, All)
- Origin region filtering (Nordic, European, International, etc.)
- Database-backed name storage
- Git submodule integration for name data updates
- Ratings persistence with SQLite

### 📊 Optimizations
- **Batch processing** for origin classification (up to 100 names at once)
- **Efficient database sync** with commit hash tracking
- **Bulk inserts** for new names
- **Comprehensive logging** (DEBUG, INFO, WARNING levels)
- **Fallback mechanisms** for error recovery

## Setup

### Prerequisites
- Python 3.13 or higher
- Git (for submodule management)

### Installation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/sort-names.git
   cd sort-names
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Optional: Install name2nat for origin classification:**
   ```bash
   pip install name2nat
   ```

### Database Initialization

The application uses SQLite (`names.db`). The database is automatically initialized on first run.

To manually initialize or sync names from the submodule:
```bash
python init_database.py
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`.

### Application Interface

1. **Sidebar**
   - Submodule management (reload names, check for updates)
   - Gender filter selection
   - Origin region filter (multiselect)
   - Ratings management (save, reset, export)

2. **Main Content**
   - **Tournament tab**: Compare two names, vote, see top rankings
   - **Similarity tab**: Search for names similar to a reference

### Origin Classification

To classify name origins (requires `name2nat`):
```bash
python classify_origins.py
```

Options:
- `--limit N`: Limit to N names (for testing)
- `--batch-size N`: Set batch size (default: 100)
- `--stats`: Show classification statistics

### Database Management

- Names are stored in `names.db` SQLite database
- Ratings are persisted automatically
- Submodule updates are tracked by commit hash
- Unclassified names can be processed in batches

## Optimization Details

### Batch Processing
- Origin classification processes up to 100 names simultaneously
- Reduces API calls to `name2nat` by ~100x
- Fallback to individual classification if batch fails

### Database Efficiency
- Bulk inserts using `executemany` for new names
- Unique constraints prevent duplicates
- Indexed columns for fast filtering
- Commit hash tracking avoids redundant processing

### Logging System
- Structured logging with timestamps and module names
- DEBUG level: Detailed operations (file loads, filtering, sync status)
- INFO level: Milestones (database initialized, names loaded, ratings saved)
- WARNING level: Potential issues (missing files, classification failures)

### Error Handling
- Graceful fallbacks for missing dependencies
- Individual name classification continues if batch fails
- Database transactions ensure data consistency

## Database Schema

```sql
-- Names table
CREATE TABLE names (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
    origin_region TEXT,
    origin_confidence REAL,
    origin_classified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ratings table
CREATE TABLE ratings (
    name_id INTEGER PRIMARY KEY REFERENCES names(id) ON DELETE CASCADE,
    rating REAL NOT NULL DEFAULT 1500.0,
    matches INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Region mapping table
CREATE TABLE region_mapping (
    nationality TEXT PRIMARY KEY,
    region TEXT NOT NULL
);

-- Source versions table (submodule tracking)
CREATE TABLE source_versions (
    id INTEGER PRIMARY KEY,
    commit_hash TEXT NOT NULL,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User settings table
CREATE TABLE user_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## Development

### Code Structure
- `main.py` - Streamlit application entry point
- `database.py` - SQLite database operations
- `data_loader.py` - Name loading and validation
- `elo.py` - Elo rating calculations
- `similarity.py` - Name similarity functions
- `ui.py` - Streamlit UI components
- `utils.py` - Utility functions
- `classify_origins.py` - Origin classification script
- `init_database.py` - Database initialization script

### Testing
Run basic functionality tests:
```bash
python verify_imports.py
```

### Code Quality
- Uses `ruff` for linting
- Follows Python type hints
- Consistent code formatting

## Data Sources

Names are sourced from the `godkendtefornavne` submodule, which contains approved Danish names from the Danish government.

## License

[Add your license here]

## Acknowledgments

- [name2nat](https://github.com/ivarvit/name2nat) for name nationality prediction
- Streamlit for the web application framework
- The Danish government for the name data
