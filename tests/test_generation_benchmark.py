import unittest

import numpy as np

from src.evaluation.generation_benchmark import (
    extract_citations,
    score_answer,
    semantic_answer_similarity,
)


class GenerationScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [{
            "chunk_id": "gold-1",
            "document_id": "guide.pdf",
            "page_start": 7,
            "page_end": 7,
            "text": "The answer is alpha.",
        }]

    def test_extracts_citations(self) -> None:
        self.assertEqual(extract_citations("Alpha [guide.pdf, PDF p. 7]."), ["guide.pdf, PDF p. 7"])

    def test_grounded_answer_uses_gold_context(self) -> None:
        result = score_answer(
            "The answer is alpha [guide.pdf, PDF p. 7].",
            self.chunks,
            ["gold-1"],
            "The answer is alpha.",
        )
        self.assertTrue(result["grounded"])
        self.assertEqual(result["citation_correctness"], 1.0)

    def test_unknown_citation_is_not_grounded(self) -> None:
        result = score_answer(
            "The answer is alpha [other.pdf, PDF p. 2].",
            self.chunks,
            ["gold-1"],
            "The answer is alpha.",
        )
        self.assertFalse(result["grounded"])
        self.assertEqual(result["citation_validity"], 0.0)

    def test_semantic_similarity_ignores_citation_text(self) -> None:
        class FakeEmbeddingModel:
            def encode(self, texts, **kwargs):
                self.texts = texts
                return np.array([[1.0, 0.0], [0.8, 0.6]])

        model = FakeEmbeddingModel()
        similarity = semantic_answer_similarity(
            "Alpha [guide.pdf, PDF p. 7].", "Alpha.", model
        )
        self.assertEqual(model.texts[0], "Alpha .")
        self.assertAlmostEqual(similarity, 0.8)


if __name__ == "__main__":
    unittest.main()
