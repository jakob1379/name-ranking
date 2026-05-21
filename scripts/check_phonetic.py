#!/usr/bin/env python3
"""
Check phonetic code population status in database.
"""

import sqlite3

from st_name_ranking.persistence.connection import get_db_path


def main() -> None:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Total names
    cursor.execute("SELECT COUNT(*) as total FROM names")
    total = cursor.fetchone()[0]

    # Names with phonetic_primary populated
    cursor.execute("""
        SELECT COUNT(*) as with_phonetic
        FROM names
        WHERE phonetic_primary IS NOT NULL AND phonetic_primary != ''
    """)
    with_phonetic = cursor.fetchone()[0]

    # Names with phonetic_secondary populated
    cursor.execute("""
        SELECT COUNT(*) as with_phonetic_secondary
        FROM names
        WHERE phonetic_secondary IS NOT NULL AND phonetic_secondary != ''
    """)
    with_phonetic_secondary = cursor.fetchone()[0]

    # Names with both populated
    cursor.execute("""
        SELECT COUNT(*) as with_both
        FROM names
        WHERE phonetic_primary IS NOT NULL AND phonetic_primary != ''
        AND phonetic_secondary IS NOT NULL AND phonetic_secondary != ''
    """)
    with_both = cursor.fetchone()[0]

    # Names with NULL or empty
    cursor.execute("""
        SELECT COUNT(*) as missing
        FROM names
        WHERE phonetic_primary IS NULL OR phonetic_primary = ''
        OR phonetic_secondary IS NULL OR phonetic_secondary = ''
    """)
    missing = cursor.fetchone()[0]

    print(f"Total names: {total}")
    print(
        f"With phonetic_primary: {with_phonetic} ({with_phonetic / total * 100:.1f}%)",
    )
    print(
        f"With phonetic_secondary: {with_phonetic_secondary} ({with_phonetic_secondary / total * 100:.1f}%)",
    )
    print(f"With both: {with_both} ({with_both / total * 100:.1f}%)")
    print(f"Missing: {missing} ({missing / total * 100:.1f}%)")

    # Show a few examples
    cursor.execute("""
        SELECT name, phonetic_primary, phonetic_secondary
        FROM names
        WHERE phonetic_primary IS NOT NULL AND phonetic_primary != ''
        LIMIT 5
    """)
    print("\nExamples:")
    for row in cursor.fetchall():
        print(
            f"  {row['name']}: primary={row['phonetic_primary']}, secondary={row['phonetic_secondary']}",
        )

    conn.close()


if __name__ == "__main__":
    main()
