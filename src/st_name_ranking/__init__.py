"""Name Ranking Application.

Canonical imports live in the domain subpackages:

- ``st_name_ranking.active_learning``
- ``st_name_ranking.classification``
- ``st_name_ranking.commands``
- ``st_name_ranking.interface``
- ``st_name_ranking.learning``
- ``st_name_ranking.persistence``

Stable package-level shared modules:

- ``st_name_ranking.types`` owns cross-package record types.
- ``st_name_ranking.persistence.name_normalization`` owns loader/persistence name cleanup.
- ``st_name_ranking.active_learning.phonetic_similarity`` owns generic phonetic scoring.

Top-level database/features/UI-style modules are deprecated compatibility shims
for older callers. Internal code should use canonical subpackage imports.
"""
