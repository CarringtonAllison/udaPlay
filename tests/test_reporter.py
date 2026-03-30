"""Tests for ReportFormatter."""
import json
import pytest


_SAMPLE_ANSWER = (
    "Elden Ring was developed by FromSoftware and published by Bandai Namco. "
    "It was released on February 25, 2022 for PlayStation, Xbox, and PC."
)

_SAMPLE_CITATIONS = [
    {"title": "Elden Ring", "url": None, "source": "internal"},
]

_SAMPLE_JSON_BLOCK = json.dumps({
    "answer_summary": "Elden Ring was made by FromSoftware.",
    "games_mentioned": ["Elden Ring"],
    "confidence": 0.95,
    "sources": [{"title": "Elden Ring", "url": None}],
})


class TestReportFormatter:

    def test_text_mode_returns_string(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        output = fmt.format(
            answer=_SAMPLE_ANSWER,
            citations=_SAMPLE_CITATIONS,
            mode="text",
        )
        assert isinstance(output, str)
        assert "Elden Ring" in output

    def test_json_mode_returns_dict(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        output = fmt.format(
            answer=_SAMPLE_ANSWER,
            citations=_SAMPLE_CITATIONS,
            mode="json",
            confidence=0.95,
        )
        assert isinstance(output, dict)
        assert "answer_summary" in output or "answer" in output
        assert "confidence" in output

    def test_both_mode_returns_dict_with_text_and_json(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        output = fmt.format(
            answer=_SAMPLE_ANSWER,
            citations=_SAMPLE_CITATIONS,
            mode="both",
            confidence=0.9,
        )
        assert isinstance(output, dict)
        assert "text" in output
        assert "json" in output
        assert isinstance(output["text"], str)
        assert isinstance(output["json"], dict)

    def test_citations_included_in_text_output(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        output = fmt.format(
            answer=_SAMPLE_ANSWER,
            citations=[{"title": "Elden Ring", "url": "https://example.com", "source": "internal"}],
            mode="text",
        )
        assert "Elden Ring" in output

    def test_web_search_citation_includes_url(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        web_citation = {"title": "IGN Review", "url": "https://ign.com/elden-ring", "source": "web_search"}
        output = fmt.format(
            answer="Elden Ring got a 10/10.",
            citations=[web_citation],
            mode="text",
        )
        assert "ign.com" in output or "IGN" in output

    def test_invalid_mode_raises_value_error(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        with pytest.raises(ValueError, match="mode"):
            fmt.format(answer="test", citations=[], mode="invalid_mode")

    def test_json_mode_includes_confidence(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        output = fmt.format(
            answer="Some answer.",
            citations=[],
            mode="json",
            confidence=0.82,
        )
        assert output["confidence"] == pytest.approx(0.82)

    def test_empty_citations_does_not_crash(self):
        from src.reporter import ReportFormatter
        fmt = ReportFormatter()
        for mode in ("text", "json", "both"):
            output = fmt.format(answer="A plain answer.", citations=[], mode=mode)
            assert output is not None
