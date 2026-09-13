"""Offline tests for the reusable news NLP pipeline."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import (  # noqa: E402
    MiniLMFinancialFallback,
    NewsPipeline,
    TransformersClassifier,
    aggregate_event_features,
    build_default_pipeline,
    build_direction_input,
    compute_tone_momentum,
    deduplicate_events,
    issuer_aliases,
    mask_company_aliases,
)


class FixedClassifier:
    """Minimal injected classifier; it records exactly what the pipeline sends."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def predict_proba(self, texts):
        self.calls.append(list(texts))
        return self.rows


class FixedFallback:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, texts):
        self.calls.append(list(texts))
        return self.scores


class TextPreparationTest(unittest.TestCase):
    def test_company_name_and_aliases_are_masked(self):
        aliases = issuer_aliases("First Solar, Inc.", "First Solar|FSLR")
        masked = mask_company_aliases("First Solar's solar project expands", aliases)
        self.assertEqual(masked, "the company solar project expands")

    def test_direction_input_matches_training_format(self):
        self.assertEqual(
            build_direction_input("Example Co", "social", "Workers approve an agreement"),
            "Company: Example Co. Pillar: social. Headline: Workers approve an agreement",
        )


class RoutingTest(unittest.TestCase):
    def test_none_only_fallback_and_assigned_only_direction(self):
        esg = FixedClassifier(
            [
                {"environmental": 0.80, "social": 0.05, "governance": 0.05, "none": 0.10},
                {"environmental": 0.05, "social": 0.05, "governance": 0.10, "none": 0.80},
                {"environmental": 0.02, "social": 0.03, "governance": 0.05, "none": 0.90},
            ]
        )
        fallback = FixedFallback([0.61, None])
        direction = FixedClassifier(
            [
                {"negative": 0.10, "neutral": 0.10, "positive": 0.80},
                {"negative": 0.70, "neutral": 0.20, "positive": 0.10},
            ]
        )
        pipeline = NewsPipeline(
            esg_classifier=esg,
            financial_fallback=fallback,
            direction_classifier=direction,
        )
        news = pd.DataFrame(
            [
                {
                    "company_id": "A",
                    "company_name": "Acme Corp",
                    "headline": "Acme cuts emissions with renewable power",
                },
                {
                    "company_id": "B",
                    "company_name": "Beta Holdings Inc.",
                    "headline": "Beta reports lower quarterly earnings",
                },
                {
                    "company_id": "C",
                    "company_name": "Gamma Inc.",
                    "headline": "Gamma sponsors a local art exhibition",
                },
            ]
        )

        scored = pipeline.classify(news, deduplicate=False)

        self.assertEqual(fallback.calls, [[news.iloc[1].headline, news.iloc[2].headline]])
        self.assertEqual(len(direction.calls[0]), 2)
        self.assertIn("Pillar: environmental", direction.calls[0][0])
        self.assertIn("Pillar: financial", direction.calls[0][1])
        self.assertNotIn("Acme", esg.calls[0][0])
        self.assertNotIn("Beta", esg.calls[0][1])

        self.assertEqual(
            scored["pillar_label"].tolist(),
            ["environmental", "financial", "unclassified"],
        )
        self.assertEqual(scored["fallback_called"].tolist(), [0, 1, 1])
        self.assertEqual(scored["financial_fallback_accepted"].tolist(), [0, 1, 0])
        self.assertAlmostEqual(scored.iloc[0].signed_tone, 0.70)
        self.assertAlmostEqual(scored.iloc[1].signed_tone, -0.60)
        self.assertTrue(math.isnan(scored.iloc[2].signed_tone))
        self.assertEqual(scored.iloc[2].inference_status, "unclassified_no_financial_evidence")

    def test_bad_probability_contract_is_rejected(self):
        pipeline = NewsPipeline(
            esg_classifier=FixedClassifier(
                [{"environmental": 0.5, "social": 0.1, "governance": 0.1, "none": 0.1}]
            ),
            financial_fallback=FixedFallback([]),
            direction_classifier=FixedClassifier([]),
        )
        news = pd.DataFrame(
            [{"company_id": "A", "company_name": "Acme", "headline": "A headline"}]
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            pipeline.classify(news, deduplicate=False)


class EventDeduplicationTest(unittest.TestCase):
    @staticmethod
    def _news():
        return pd.DataFrame(
            [
                {
                    "event_id": "source-1",
                    "company_id": "A",
                    "company_name": "Acme",
                    "headline": "Acme cuts carbon emissions with new renewable energy plan",
                    "published_at_utc": "2026-01-01T00:00:00Z",
                    "domains": "one.example",
                    "article_count": 1,
                    "relevance_strength": 0.8,
                },
                {
                    "event_id": "source-2",
                    "company_id": "A",
                    "company_name": "Acme",
                    "headline": "Acme cuts carbon emissions with renewable energy plan",
                    "published_at_utc": "2026-01-01T02:00:00Z",
                    "domains": "two.example",
                    "article_count": 2,
                    "relevance_strength": 0.9,
                },
                {
                    "event_id": "source-3",
                    "company_id": "A",
                    "company_name": "Acme",
                    "headline": "Acme cuts carbon emissions with renewable energy plan",
                    "published_at_utc": "2026-01-04T08:00:01Z",
                    "domains": "three.example",
                    "article_count": 1,
                    "relevance_strength": 0.9,
                },
                {
                    "event_id": "source-4",
                    "company_id": "B",
                    "company_name": "Beta",
                    "headline": "Acme cuts carbon emissions with renewable energy plan",
                    "published_at_utc": "2026-01-01T01:00:00Z",
                    "domains": "four.example",
                    "article_count": 1,
                    "relevance_strength": 0.7,
                },
            ]
        )

    def test_dedup_is_company_specific_and_deterministic(self):
        forward = deduplicate_events(self._news())
        reverse = deduplicate_events(self._news().iloc[::-1].reset_index(drop=True))

        self.assertEqual(len(forward), 3)
        pd.testing.assert_frame_equal(forward, reverse)
        combined = forward.loc[forward["near_duplicate_member_count"].eq(2)].iloc[0]
        self.assertEqual(combined.article_count, 3)
        self.assertEqual(combined.source_count, 2)
        self.assertEqual(combined.domains, "one.example|two.example")
        self.assertEqual(combined.exact_event_ids, "source-1|source-2")
        self.assertAlmostEqual(combined.relevance_strength, 0.9)


class AggregationTest(unittest.TestCase):
    def test_event_id_is_required(self):
        scored = pd.DataFrame(
            [{
                "company_id": "A",
                "pillar_label": "social",
                "published_at_utc": "2026-01-30T00:00:00Z",
                "signed_tone": 0.2,
            }]
        )
        with self.assertRaisesRegex(ValueError, "event_id"):
            aggregate_event_features(scored, ["A"], "2026-01-31T00:00:00Z")

    def test_duplicate_company_event_is_rejected(self):
        scored = pd.DataFrame(
            [{
                "company_id": "A",
                "event_id": "same-event",
                "pillar_label": "social",
                "published_at_utc": "2026-01-30T00:00:00Z",
                "signed_tone": 0.2,
            }]
            * 2
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            aggregate_event_features(scored, ["A"], "2026-01-31T00:00:00Z")

    def test_windows_weights_shrinkage_and_missingness(self):
        scored = pd.DataFrame(
            [
                {
                    "company_id": "A",
                    "event_id": "one",
                    "pillar_label": "environmental",
                    "published_at_utc": "2026-01-26T00:00:00Z",
                    "signed_tone": 0.6,
                    "direction_label": "positive",
                    "p_negative": 0.1,
                    "p_neutral": 0.2,
                    "p_positive": 0.7,
                    "esg_top_probability": 0.8,
                    "financial_fallback_score": float("nan"),
                    "direction_confidence": 0.7,
                    "relevance_strength": 0.9,
                    "domains": "one.example|two.example",
                    "article_count": 2,
                },
                {
                    "company_id": "A",
                    "event_id": "two",
                    "pillar_label": "environmental",
                    "published_at_utc": "2025-12-27T00:00:00Z",
                    "signed_tone": -0.4,
                    "direction_label": "negative",
                    "p_negative": 0.6,
                    "p_neutral": 0.2,
                    "p_positive": 0.2,
                    "esg_top_probability": 0.75,
                    "financial_fallback_score": float("nan"),
                    "direction_confidence": 0.4,
                    "relevance_strength": 0.8,
                    "domains": "three.example",
                    "article_count": 1,
                },
            ]
        )

        features = aggregate_event_features(
            scored,
            company_ids=["A", "B"],
            as_of_utc="2026-01-31T00:00:00Z",
        )

        self.assertEqual(len(features), 2 * 4 * 3)
        row_30 = features.query(
            "company_id == 'A' and pillar == 'environmental' and window_days == 30"
        ).iloc[0]
        expected_weight = 0.9 * (0.5 ** (5 / 30))
        self.assertEqual(row_30.event_count, 1)
        self.assertAlmostEqual(row_30.effective_weight, expected_weight)
        self.assertAlmostEqual(row_30.tone_raw, 0.6)
        self.assertAlmostEqual(row_30.tone_shrunk_to_zero, 0.6 * expected_weight / 3)

        row_90 = features.query(
            "company_id == 'A' and pillar == 'environmental' and window_days == 90"
        ).iloc[0]
        self.assertEqual(row_90.event_count, 2)
        self.assertEqual(row_90.source_count, 3)
        self.assertLess(row_90.tone_raw, 0.6)

        no_news = features.query(
            "company_id == 'B' and pillar == 'environmental' and window_days == 90"
        ).iloc[0]
        self.assertEqual(no_news.has_evidence, 0)
        self.assertTrue(math.isnan(no_news.tone_raw))
        self.assertEqual(no_news.tone_shrunk_to_zero, 0.0)

        momentum = compute_tone_momentum(features)
        observed = momentum.query(
            "company_id == 'A' and pillar == 'environmental'"
        ).iloc[0]
        self.assertAlmostEqual(
            observed.tone_momentum,
            row_30.tone_shrunk_to_zero - row_90.tone_shrunk_to_zero,
        )

        no_news_momentum = momentum.query(
            "company_id == 'B' and pillar == 'environmental'"
        ).iloc[0]
        self.assertEqual(no_news_momentum.tone_momentum, 0.0)
        self.assertEqual(no_news_momentum.baseline_window_has_evidence, 0)


class LazyLoadingTest(unittest.TestCase):
    def test_default_pipeline_does_not_load_or_download_models(self):
        pipeline = build_default_pipeline("a/local/checkpoint")
        self.assertIsInstance(pipeline.esg_classifier, TransformersClassifier)
        self.assertIsInstance(pipeline.financial_fallback, MiniLMFinancialFallback)
        self.assertIsNone(pipeline.esg_classifier._model)
        self.assertIsNone(pipeline.financial_fallback._model)
        self.assertIsNone(pipeline.direction_classifier._model)
        self.assertEqual(
            pipeline.esg_classifier.revision,
            "f79fefa034aa8a969379e23b755369a94c4cd0d3",
        )
        self.assertEqual(
            pipeline.financial_fallback.revision,
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        )


if __name__ == "__main__":
    unittest.main()
