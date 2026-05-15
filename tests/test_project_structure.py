"""
tests/test_project_structure.py — Lightweight project structure validation.

Checks that required files exist, key CSVs are readable and correctly shaped,
and required columns are present. Does not call external APIs or rerun the pipeline.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# ── Required file list ────────────────────────────────────────────────────────

REQUIRED_FILES = [
    # Entry point
    "app.py",
    "requirements.txt",
    "README.md",
    # Dashboard source
    "src/__init__.py",
    "src/dashboard/__init__.py",
    "src/dashboard/data_loader.py",
    "src/dashboard/charts.py",
    "src/dashboard/components.py",
    "src/dashboard/evidence_cards.py",
    "src/processing/__init__.py",
    # Processed outputs — must be committed for CI to pass
    "Data/processed/research_gap_scores.csv",
    "Data/processed/target_evidence_cards.csv",
    "Data/processed/knowledge_graph_nodes.csv",
    "Data/processed/knowledge_graph_edges.csv",
    # Reports
    "outputs/reports/gap_scoring_summary.md",
    "outputs/reports/knowledge_graph_summary.md",
]


# ── File existence checks ─────────────────────────────────────────────────────

@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_file_exists(rel_path: str):
    """Every file in REQUIRED_FILES must be present in the repository."""
    assert (ROOT / rel_path).exists(), (
        f"Required file not found: {rel_path}\n"
        "Commit all pipeline outputs before pushing."
    )


# ── research_gap_scores.csv ───────────────────────────────────────────────────

class TestGapScores:
    """Validate the primary gap-scoring output table."""

    _PATH = ROOT / "Data/processed/research_gap_scores.csv"

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(self._PATH)

    def test_readable(self, df):
        """CSV must load without error."""
        assert df is not None

    def test_not_empty(self, df):
        """Table must contain at least one row."""
        assert len(df) > 0, "research_gap_scores.csv is empty"

    def test_row_count(self, df):
        """Expected 499 scored targets (one per Open Targets entry)."""
        assert len(df) == 499, (
            f"Expected 499 rows in research_gap_scores.csv, got {len(df)}"
        )

    @pytest.mark.parametrize("col", [
        "approved_symbol",
        "approved_name",
        "association_score",
        "gap_score",
        "gap_interpretation_tier",
        "literature_mentions",
        "trial_mentions",
        "symbol_ambiguity_flag",
    ])
    def test_required_column_present(self, df, col: str):
        """All scoring columns must be present."""
        assert col in df.columns, (
            f"Missing required column '{col}' in research_gap_scores.csv"
        )

    def test_gap_score_range(self, df):
        """Gap scores must be finite and within [0, 1]."""
        scores = df["gap_score"].dropna()
        assert (scores >= 0).all() and (scores <= 1).all(), (
            "gap_score values outside expected [0, 1] range"
        )

    def test_no_duplicate_symbols(self, df):
        """Each target symbol should appear at most once."""
        dupes = df["approved_symbol"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate approved_symbol entries found"


# ── target_evidence_cards.csv ────────────────────────────────────────────────

class TestEvidenceCards:
    """Validate the evidence card output table."""

    _PATH = ROOT / "Data/processed/target_evidence_cards.csv"

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(self._PATH)

    def test_readable(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0, "target_evidence_cards.csv is empty"

    def test_row_count(self, df):
        """One evidence card per scored target (499 expected)."""
        assert len(df) >= 499, (
            f"Expected >= 499 evidence cards, got {len(df)}"
        )

    @pytest.mark.parametrize("col", [
        "approved_symbol",
        "gap_interpretation_tier",
        "evidence_strength_label",
        "literature_signal_label",
        "trial_signal_label",
        "evidence_card_path",
    ])
    def test_required_column_present(self, df, col: str):
        assert col in df.columns, (
            f"Missing required column '{col}' in target_evidence_cards.csv"
        )

    def test_all_cards_have_path(self, df):
        """Every target must have a non-null evidence card path."""
        missing = df["evidence_card_path"].isna().sum()
        assert missing == 0, f"{missing} targets have no evidence_card_path"


# ── knowledge_graph_nodes.csv ────────────────────────────────────────────────

class TestGraphNodes:
    """Validate the knowledge graph node table."""

    _PATH = ROOT / "Data/processed/knowledge_graph_nodes.csv"

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(self._PATH)

    def test_readable(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0, "knowledge_graph_nodes.csv is empty"

    @pytest.mark.parametrize("col", ["node_id", "node_type", "label"])
    def test_required_column_present(self, df, col: str):
        assert col in df.columns, (
            f"Missing required column '{col}' in knowledge_graph_nodes.csv"
        )

    def test_no_null_node_ids(self, df):
        """node_id must be fully populated — it is the graph primary key."""
        nulls = df["node_id"].isna().sum()
        assert nulls == 0, f"{nulls} null node_id values found"

    def test_expected_node_types(self, df):
        """Graph must contain all six expected node types."""
        expected = {"Disease", "Target", "Paper", "Trial", "LiteratureTheme", "TrialTheme"}
        actual   = set(df["node_type"].dropna().unique())
        missing  = expected - actual
        assert not missing, f"Missing node types in graph: {missing}"


# ── knowledge_graph_edges.csv ────────────────────────────────────────────────

class TestGraphEdges:
    """Validate the knowledge graph edge table."""

    _PATH = ROOT / "Data/processed/knowledge_graph_edges.csv"

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(self._PATH)

    def test_readable(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0, "knowledge_graph_edges.csv is empty"

    @pytest.mark.parametrize("col", ["source", "target", "edge_type"])
    def test_required_column_present(self, df, col: str):
        assert col in df.columns, (
            f"Missing required column '{col}' in knowledge_graph_edges.csv"
        )

    def test_no_null_source_or_target(self, df):
        """Every edge must have both a source and a target node."""
        null_src = df["source"].isna().sum()
        null_tgt = df["target"].isna().sum()
        assert null_src == 0 and null_tgt == 0, (
            f"Null values in edge endpoints: source={null_src}, target={null_tgt}"
        )

    def test_expected_edge_types(self, df):
        """Graph must contain disease-target and mention edge types."""
        expected = {
            "disease_target_evidence",
            "target_mentioned_in_paper",
            "paper_has_literature_theme",
        }
        actual  = set(df["edge_type"].dropna().unique())
        missing = expected - actual
        assert not missing, f"Missing edge types in graph: {missing}"


# ── Evidence card markdown files ─────────────────────────────────────────────

class TestEvidenceCardFiles:
    """Validate that key individual markdown evidence card files exist."""

    SPOT_CHECK_SYMBOLS = ["TREM2", "APOE", "APP", "CD33", "ABHD6"]

    @pytest.mark.parametrize("sym", SPOT_CHECK_SYMBOLS)
    def test_markdown_card_exists(self, sym: str):
        """Spot-check that a markdown card was generated for key targets."""
        card_path = ROOT / "outputs" / "evidence_cards" / f"{sym}_evidence_card.md"
        assert card_path.exists(), (
            f"Markdown evidence card not found for {sym}: {card_path}"
        )

    @pytest.mark.parametrize("sym", SPOT_CHECK_SYMBOLS)
    def test_markdown_card_not_empty(self, sym: str):
        card_path = ROOT / "outputs" / "evidence_cards" / f"{sym}_evidence_card.md"
        if card_path.exists():
            content = card_path.read_text(encoding="utf-8")
            assert len(content) > 100, f"Evidence card for {sym} appears too short"

    @pytest.mark.parametrize("sym", SPOT_CHECK_SYMBOLS)
    def test_markdown_card_has_no_clinical_claims(self, sym: str):
        """No card must contain forbidden clinical overclaim phrases."""
        card_path = ROOT / "outputs" / "evidence_cards" / f"{sym}_evidence_card.md"
        if not card_path.exists():
            pytest.skip(f"Card not found for {sym}")
        text = card_path.read_text(encoding="utf-8").lower()
        forbidden = ["therapeutically valid", "guaranteed", "proves absence"]
        for phrase in forbidden:
            assert phrase not in text, (
                f"Forbidden phrase '{phrase}' found in {sym}_evidence_card.md"
            )
