"""Shared pytest fixtures.

Every fixture here builds objects against temporary directories, so the test
suite never touches the real corpus on the shared drive nor the project's own
``data/`` index. The corpus fixtures create a small synthetic tree that mimics
the shape of the real one — nested folders, mixed formats, an excluded
developer directory — without needing any real accreditation documents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EnvironmentReader, Settings  # noqa: E402
from app.safety import SourceTreeGuard  # noqa: E402


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A Settings object pointing at throwaway source and data directories.

    Environment variables are cleared first so a developer's own ``.env`` cannot
    change what the tests assert.
    """
    # Cleared so the suite asserts the *defaults*, never whatever the developer
    # happens to have in their own .env — python-dotenv has already loaded that
    # file into the environment by the time these tests run.
    for variable_name in (
        "SOURCE_DIR", "EXTRA_SOURCE_DIRS", "DATA_DIR", "EXCLUDE_DIRS", "TEXT_EXTENSIONS",
        "CHUNK_TOKENS", "CHUNK_OVERLAP_TOKENS", "MIN_CHUNK_CHARS",
        "TOP_K", "RRF_K", "MAX_CONTEXT_CHARS", "LLM_BACKEND",
        "OCR_ENABLED", "RERANK_ENABLED", "RERANK_BACKEND", "RERANK_OPENVINO_DIR",
        "QUERY_EMBED_DEVICE", "QUERY_EMBED_BACKEND", "QUERY_EMBED_OPENVINO_DIR", "PRELOAD_MODELS",
        "EMBED_DIM", "MAX_FILE_MB",
        "HOST", "PORT", "ALLOW_OPEN_FILE",
        "HISTORY_TURNS", "MAX_HISTORY_CHARS", "FOLLOW_UP_WORDS",
        "CLAUDE_CLI_MODEL", "CLAUDE_CLI_EFFORT", "CLAUDE_CLI_TIMEOUT",
        "CODEX_CLI_MODEL", "CODEX_CLI_EFFORT", "CODEX_CLI_TIMEOUT",
        "ANTIGRAVITY_CLI_MODEL", "ANTIGRAVITY_CLI_EFFORT", "ANTIGRAVITY_CLI_TIMEOUT",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    settings = Settings(EnvironmentReader())
    settings.source_directory = tmp_path / "corpus"
    settings.data_directory = tmp_path / "index-data"
    # Cleared deliberately: the default points at the project's real
    # `corpus-lokal`, which the suite must never read or depend on.
    settings.additional_source_directories = []
    # An empty location, so no test ever loads the real converted reranker.
    settings.reranking_openvino_directory = tmp_path / "openvino-reranker" / "int8"
    settings.query_embedding_openvino_directory = tmp_path / "openvino-embedder" / "int8"
    settings.source_directory.mkdir(parents=True, exist_ok=True)
    settings.create_directories()
    return settings


@pytest.fixture
def extra_corpus_root(isolated_settings: Settings, tmp_path: Path) -> Path:
    """A second, non-primary corpus root registered on the settings.

    Mirrors the real ``corpus-lokal`` directory: material copied out of the
    shared drive and indexed alongside it.
    """
    extra_root = tmp_path / "corpus-lokal"
    extra_root.mkdir(parents=True, exist_ok=True)
    (extra_root / "data_dosen_sinta.xlsx.txt").write_text(
        "nama_dosen\tgoogle_scholar\tsinta\n"
        "RICHARDUS EKO INDRAJIT\thttps://scholar.google.com/citations?user=X\t"
        "https://sinta.kemdiktisaintek.go.id/authors/profile/1",
        encoding="utf-8",
    )
    isolated_settings.additional_source_directories = [extra_root]
    return extra_root


@pytest.fixture
def guard(isolated_settings: Settings) -> SourceTreeGuard:
    """A read-only guard bound to the isolated settings."""
    return SourceTreeGuard(isolated_settings)


@pytest.fixture
def sample_corpus(isolated_settings: Settings) -> Path:
    """A miniature corpus covering the cases the scanner must handle.

    Layout::

        corpus/
          1. Budaya Mutu/catatan.txt      indexable, nested
          1. Budaya Mutu/data.csv         indexable, tabular
          laporan.md                      indexable, at the root
          config.json                     indexable
          rekaman.mov                     skipped by extension
          pintasan.lnk                    skipped by extension
          zdev/script.py                  skipped by excluded directory
    """
    root = isolated_settings.source_directory
    nested = root / "1. Budaya Mutu"
    nested.mkdir(parents=True, exist_ok=True)

    (nested / "catatan.txt").write_text(
        "Beban mengajar dosen tetap program studi diukur dengan EWMP.\n"
        "Rata-rata EWMP pada TS adalah 12,5 SKS per semester.",
        encoding="utf-8",
    )
    (nested / "data.csv").write_text(
        "Nama,Jabatan,EWMP\nGufron,Lektor,13\nAndi,Asisten Ahli,11\n",
        encoding="utf-8",
    )
    (root / "laporan.md").write_text(
        "# Laporan Akreditasi\n\nDokumen ini memuat capaian program studi.",
        encoding="utf-8",
    )
    (root / "config.json").write_text('{"tahun": 2026, "prodi": "MTI"}', encoding="utf-8")
    (root / "rekaman.mov").write_bytes(b"\x00" * 64)
    (root / "pintasan.lnk").write_bytes(b"\x00" * 32)

    developer_directory = root / "zdev"
    developer_directory.mkdir(exist_ok=True)
    (developer_directory / "script.py").write_text("print('bukan dokumen')", encoding="utf-8")

    return root
