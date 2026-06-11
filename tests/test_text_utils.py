import unittest

from rag import (
    extract_chinese_terms,
    extract_english_words,
    highlight_text,
    normalize_keyword_score,
    normalize_text,
    parse_keywords,
    score_paragraph,
)


class TextUtilityTests(unittest.TestCase):
    def test_normalize_text_lowercases_and_collapses_spaces(self):
        self.assertEqual(normalize_text("  RAG   System  "), "rag system")

    def test_extract_english_words(self):
        self.assertEqual(extract_english_words("RAG system 101!"), ["rag", "system", "101"])

    def test_extract_chinese_terms(self):
        self.assertEqual(extract_chinese_terms("RAG 是 檢索增強生成。"), ["是", "檢索增強生成"])

    def test_parse_keywords_combines_english_and_chinese_terms(self):
        self.assertEqual(parse_keywords("RAG 是什麼 embedding"), ["rag", "embedding", "是什麼"])

    def test_score_paragraph_counts_direct_matches_and_deduplicates_keywords(self):
        score, matched = score_paragraph("RAG helps RAG retrieval", ["rag", "retrieval"])
        self.assertEqual(score, 9)
        self.assertEqual(matched, ["rag", "retrieval"])

    def test_normalize_keyword_score_caps_at_one(self):
        self.assertEqual(normalize_keyword_score(0), 0.0)
        self.assertEqual(normalize_keyword_score(3), 0.5)
        self.assertEqual(normalize_keyword_score(99), 1.0)

    def test_highlight_text_wraps_matches_with_ansi_escape(self):
        highlighted = highlight_text("RAG retrieval", ["rag"])
        self.assertIn("\033[30;43mRAG\033[0m", highlighted)


if __name__ == "__main__":
    unittest.main()
