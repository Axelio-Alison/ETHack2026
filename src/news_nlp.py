"""Reusable building blocks for the company-news ESG prototype.

The module keeps model loading out of notebooks.  It also accepts injected
predictors, which makes the complete routing logic testable without internet
access or large model downloads.

The input to :class:`NewsPipeline` is already expected to have passed the
separate company-relevance gate.  ``relevance_strength`` is evidence strength,
not a calibrated probability.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


PILLARS = ("financial", "environmental", "social", "governance")
ESG_LABELS = ("environmental", "social", "governance", "none")
DIRECTION_LABELS = ("negative", "neutral", "positive")

DEFAULT_ESG_MODEL = "yiyanghkust/finbert-esg"
DEFAULT_FINANCIAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ESG_REVISION = "f79fefa034aa8a969379e23b755369a94c4cd0d3"
DEFAULT_FINANCIAL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

FINANCIAL_KEYWORDS = (
    "acquisition",
    "bankruptcy",
    "cash flow",
    "credit",
    "debt",
    "default",
    "dividend",
    "earnings",
    "forecast",
    "funding",
    "guidance",
    "investment",
    "margin",
    "merger",
    "net loss",
    "operating loss",
    "posts a loss",
    "profit",
    "quarterly loss",
    "restructuring",
    "revenue",
    "sales",
    "shares",
    "stock",
    "write-down",
    "writedown",
)

FINANCIAL_PROTOTYPES = (
    "The company reports earnings, revenue, profit, cash flow, debt or financial guidance.",
    "This business event affects financial resilience, solvency, investment or shareholder value.",
    "The firm announces a merger, acquisition, restructuring, financing or bankruptcy.",
)

_STEM_KEYWORDS = {"emission", "recycl"}
_LEGAL_SUFFIX_RE = re.compile(
    r"(?:,?\s+|\s+)(?:incorporated|inc|corporation|corp|company|co|plc|"
    r"limited|ltd|holdings?|group|n\.?v\.?|s\.?a\.?|a\.?g\.?|s\.?e\.?|"
    r"l\.?p\.?|l\.?l\.?c\.?)\.?$",
    flags=re.IGNORECASE,
)


class ProbabilityClassifier(Protocol):
    """Small interface shared by the two sequence classifiers."""

    def predict_proba(self, texts: Sequence[str]) -> Sequence[Mapping[str, float]]:
        """Return one label-to-probability mapping per input text."""


class FinancialFallback(Protocol):
    """Interface for the conservative financial fallback."""

    def score(self, texts: Sequence[str]) -> Sequence[float | None]:
        """Return evidence strength, or ``None`` when evidence is insufficient."""


def normalize_text(value: str) -> str:
    """Normalize text for matching and deterministic event comparison."""

    value = html.unescape(str(value or "")).replace("&", " and ")
    value = re.sub(r"[^\w$]+", " ", value.casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _contains_term(normalized_text: str, normalized_term: str) -> bool:
    return f" {normalized_term} " in f" {normalized_text} "


def keyword_hits(text: str, terms: Iterable[str]) -> list[str]:
    """Find whole-term keyword matches, with limited stemming where intended."""

    normalized = normalize_text(text)
    hits: set[str] = set()
    for term in terms:
        normalized_term = normalize_text(term)
        if normalized_term in _STEM_KEYWORDS:
            if re.search(rf"\b{re.escape(normalized_term)}\w*\b", normalized):
                hits.add(term)
        elif _contains_term(normalized, normalized_term):
            hits.add(term)
    return sorted(hits)


def _strip_legal_suffix(value: str) -> str:
    value = str(value or "").strip()
    previous = None
    while value and value != previous:
        previous = value
        value = _LEGAL_SUFFIX_RE.sub("", value).strip(" ,.-")
    return value[4:].strip() if value.casefold().startswith("the ") else value


def _split_values(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def issuer_aliases(company_name: str, aliases: object = None) -> list[str]:
    """Combine supplied aliases with safe forms derived from the company name."""

    values = set(_split_values(aliases))
    values.add(str(company_name))
    values.add(_strip_legal_suffix(company_name))
    return sorted(
        (value for value in values if len(normalize_text(value)) >= 4),
        key=len,
        reverse=True,
    )


def mask_company_aliases(text: str, aliases: Iterable[str]) -> str:
    """Mask issuer names so they cannot accidentally determine an ESG topic."""

    masked = str(text)
    for alias in sorted(set(aliases), key=len, reverse=True):
        words = re.findall(r"\w+", alias, flags=re.UNICODE)
        if not words:
            continue
        pattern = r"\b" + r"\W+".join(re.escape(word) for word in words)
        pattern += r"\b(?:['’]s)?"
        masked = re.sub(pattern, " the company ", masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip()


def build_direction_input(company_name: str, pillar: str, headline: str) -> str:
    """Build the exact text format used while fine-tuning the direction model."""

    return f"Company: {company_name}. Pillar: {pillar}. Headline: {headline}"


def signed_tone(probabilities: Mapping[str, float]) -> float:
    """Map direction probabilities to the continuous interval [-1, 1]."""

    return float(probabilities["positive"] - probabilities["negative"])


def direction_confidence(probabilities: Mapping[str, float]) -> float:
    """Return one minus normalized three-class entropy."""

    values = np.asarray([probabilities[label] for label in DIRECTION_LABELS], dtype=float)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum()
    return float(1.0 - entropy / np.log(len(DIRECTION_LABELS)))


def _probability_rows(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    expected_count: int,
) -> list[dict[str, float]]:
    """Validate model output once, close to the model boundary."""

    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, received {len(rows)}.")

    clean_rows: list[dict[str, float]] = []
    for row in rows:
        normalized = {str(key).casefold(): float(value) for key, value in row.items()}
        if not set(labels).issubset(normalized):
            raise ValueError(f"Prediction labels must include {list(labels)}.")
        clean = {label: normalized[label] for label in labels}
        values = np.asarray(list(clean.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Probabilities must be finite and non-negative.")
        if not np.isclose(values.sum(), 1.0, atol=1e-5):
            raise ValueError("Each probability row must sum to one.")
        clean_rows.append(clean)
    return clean_rows


class TransformersClassifier:
    """Lazy Hugging Face sequence classifier with a tiny common interface."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        labels: Sequence[str],
        *,
        revision: str | None = None,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        max_length: int = 128,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.labels = tuple(label.casefold() for label in labels)
        self.revision = revision
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._id_to_label: dict[int, str] = {}

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        load_options = {"trust_remote_code": False}
        if self.revision:
            load_options["revision"] = self.revision
        if self.cache_dir:
            load_options["cache_dir"] = self.cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **load_options)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, **load_options
        ).to(self.device)
        self._torch = torch
        self._id_to_label = {
            int(index): str(label).casefold()
            for index, label in self._model.config.id2label.items()
        }
        if set(self._id_to_label.values()) != set(self.labels):
            raise ValueError(
                f"Model labels {sorted(self._id_to_label.values())} do not match "
                f"expected labels {sorted(self.labels)}."
            )

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        texts = [str(text) for text in texts]
        if not texts:
            return []
        self._load()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None

        rows: list[dict[str, float]] = []
        self._model.eval()
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self._tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                probabilities = self._torch.softmax(self._model(**encoded).logits, dim=-1)
                for vector in probabilities.detach().cpu().numpy():
                    rows.append({
                        self._id_to_label[index]: float(value)
                        for index, value in enumerate(vector)
                    })
        return _probability_rows(rows, self.labels, len(texts))


class MiniLMFinancialFallback:
    """Lazy, conservative financial check used only after FinBERT-ESG says None."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_FINANCIAL_MODEL,
        *,
        revision: str | None = DEFAULT_FINANCIAL_REVISION,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        acceptance_threshold: float = 0.46,
        keyword_boost: float = 0.055,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.revision = revision
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.acceptance_threshold = acceptance_threshold
        self.keyword_boost = keyword_boost
        self._model = None
        self._prototype_vectors = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        options = {"device": self.device, "trust_remote_code": False}
        if self.revision:
            options["revision"] = self.revision
        if self.cache_dir:
            options["cache_folder"] = self.cache_dir
        self._model = SentenceTransformer(self.model_name_or_path, **options)
        self._prototype_vectors = self._model.encode(
            list(FINANCIAL_PROTOTYPES),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def score(self, texts: Sequence[str]) -> list[float | None]:
        texts = [str(text) for text in texts]
        if not texts:
            return []
        self._load()
        assert self._model is not None and self._prototype_vectors is not None

        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores: list[float | None] = []
        for text, vector in zip(texts, vectors):
            hits = keyword_hits(text, FINANCIAL_KEYWORDS)
            similarity = float((self._prototype_vectors @ vector).max())
            strength = similarity + min(len(hits), 3) * self.keyword_boost
            scores.append(strength if hits or strength >= self.acceptance_threshold else None)
        return scores


def _event_id(company_id: str, headline: str, published_at_utc: object = "") -> str:
    payload = f"{company_id}|{normalize_text(headline)}|{published_at_utc}"
    return "EVT_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _utc_string(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _prepare_news(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"company_id", "company_name", "headline"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required news columns: {missing}")

    work = frame.copy().reset_index(drop=True)
    if "event_id" not in work:
        published = work.get("published_at_utc", pd.Series("", index=work.index))
        work["event_id"] = [
            _event_id(company_id, headline, timestamp)
            for company_id, headline, timestamp in zip(
                work["company_id"], work["headline"], published
            )
        ]
    if "aliases" not in work:
        work["aliases"] = ""
    if "relevance_strength" not in work:
        work["relevance_strength"] = 1.0
    if "domains" not in work:
        work["domains"] = ""
    if "article_count" not in work:
        work["article_count"] = 1
    return work


def _headline_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def deduplicate_events(
    frame: pd.DataFrame,
    *,
    similarity_threshold: float = 0.85,
    max_gap_hours: float = 72.0,
) -> pd.DataFrame:
    """Collapse near-identical issuer headlines using deterministic anchored clusters."""

    if "published_at_utc" not in frame:
        raise ValueError("published_at_utc is required for event deduplication.")
    if not 0 <= similarity_threshold <= 1 or max_gap_hours < 0:
        raise ValueError("Deduplication thresholds are outside their valid range.")

    work = _prepare_news(frame)
    if work.empty:
        work["first_published_at_utc"] = pd.Series(dtype="object")
        work["last_published_at_utc"] = pd.Series(dtype="object")
        work["exact_event_ids"] = pd.Series(dtype="object")
        work["near_duplicate_member_count"] = pd.Series(dtype="int64")
        work["source_count"] = pd.Series(dtype="int64")
        return work
    work["_published"] = pd.to_datetime(work["published_at_utc"], utc=True, errors="coerce")
    if work["_published"].isna().any():
        raise ValueError("published_at_utc contains an invalid timestamp.")
    work["_normalized_headline"] = work["headline"].map(normalize_text)
    work["_source_event_id"] = work["event_id"].astype(str)
    work = work.sort_values(
        ["company_id", "_published", "_normalized_headline", "_source_event_id"]
    ).reset_index(drop=True)

    assignments: dict[int, str] = {}
    for company_id, company_rows in work.groupby("company_id", sort=True):
        clusters: list[dict[str, object]] = []
        for row_index, row in company_rows.iterrows():
            tokens = _headline_tokens(row["headline"])
            candidates: list[tuple[float, str]] = []
            for cluster in clusters:
                age_hours = (row["_published"] - cluster["start"]).total_seconds() / 3600
                if age_hours < 0 or age_hours > max_gap_hours:
                    continue
                union = tokens | cluster["tokens"]
                similarity = len(tokens & cluster["tokens"]) / len(union) if union else 1.0
                if similarity >= similarity_threshold:
                    candidates.append((similarity, str(cluster["event_id"])))

            if candidates:
                selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
            else:
                selected = _event_id(company_id, row["headline"], _utc_string(row["_published"]))
                clusters.append(
                    {"event_id": selected, "start": row["_published"], "tokens": tokens}
                )
            assignments[row_index] = selected

    work["event_id"] = [assignments[index] for index in work.index]
    collapsed: list[dict[str, object]] = []
    for event_id, group in work.groupby("event_id", sort=True):
        representative = (
            group.assign(_headline_length=group["headline"].astype(str).str.len())
            .sort_values(
                ["_headline_length", "_published", "_source_event_id"],
                ascending=[False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        domains = sorted(
            {domain for packed in group["domains"] for domain in _split_values(packed)}
        )
        representative.update(
            {
                "event_id": event_id,
                "published_at_utc": _utc_string(group["_published"].min()),
                "first_published_at_utc": _utc_string(group["_published"].min()),
                "last_published_at_utc": _utc_string(group["_published"].max()),
                "exact_event_ids": "|".join(sorted(group["_source_event_id"].unique())),
                "near_duplicate_member_count": int(len(group)),
                "article_count": int(
                    pd.to_numeric(group["article_count"], errors="coerce").fillna(1).sum()
                ),
                "domains": "|".join(domains),
                "source_count": len(domains),
                "relevance_strength": float(
                    pd.to_numeric(group["relevance_strength"], errors="raise").max()
                ),
            }
        )
        temporary_columns = (
            "_published", "_normalized_headline", "_source_event_id", "_headline_length"
        )
        for temporary in temporary_columns:
            representative.pop(temporary, None)
        collapsed.append(representative)

    return pd.DataFrame(collapsed).sort_values(
        ["company_id", "published_at_utc", "event_id"]
    ).reset_index(drop=True)


class NewsPipeline:
    """Route relevant issuer news through pillar, fallback, and direction models."""

    def __init__(
        self,
        *,
        esg_classifier: ProbabilityClassifier,
        financial_fallback: FinancialFallback,
        direction_classifier: ProbabilityClassifier,
    ) -> None:
        self.esg_classifier = esg_classifier
        self.financial_fallback = financial_fallback
        self.direction_classifier = direction_classifier

    def classify(self, news: pd.DataFrame, *, deduplicate: bool = True) -> pd.DataFrame:
        """Return event-level pillar and direction decisions with full probabilities."""

        events = deduplicate_events(news) if deduplicate else _prepare_news(news)
        aliases = [
            issuer_aliases(company_name, supplied)
            for company_name, supplied in zip(events["company_name"], events["aliases"])
        ]
        events["classification_text"] = [
            mask_company_aliases(headline, row_aliases)
            for headline, row_aliases in zip(events["headline"], aliases)
        ]

        esg_rows = _probability_rows(
            self.esg_classifier.predict_proba(events["classification_text"].tolist()),
            ESG_LABELS,
            len(events),
        )
        for label in ESG_LABELS:
            events[f"esg_p_{label}"] = [row[label] for row in esg_rows]
        events["esg_top_label"] = [max(ESG_LABELS, key=row.get) for row in esg_rows]
        events["esg_top_probability"] = [max(row.values()) for row in esg_rows]
        events["pillar_label"] = events["esg_top_label"]
        events["pillar_method"] = "finbert_esg"
        events["fallback_called"] = 0
        events["financial_keyword_hit"] = 0
        events["financial_fallback_score"] = np.nan
        events["financial_fallback_accepted"] = 0
        events["inference_status"] = "pillar_assigned_direction_pending"

        # The fallback is deliberately gated: it sees only FinBERT-ESG None rows.
        none_indices = events.index[events["esg_top_label"].eq("none")].tolist()
        if none_indices:
            none_headlines = events.loc[none_indices, "headline"].astype(str).tolist()
            fallback_scores = list(self.financial_fallback.score(none_headlines))
            if len(fallback_scores) != len(none_indices):
                raise ValueError("Financial fallback returned the wrong number of scores.")
            for row_index, headline, score in zip(none_indices, none_headlines, fallback_scores):
                events.at[row_index, "fallback_called"] = 1
                events.at[row_index, "financial_keyword_hit"] = int(
                    bool(keyword_hits(headline, FINANCIAL_KEYWORDS))
                )
                if score is not None and np.isfinite(float(score)):
                    events.at[row_index, "pillar_label"] = "financial"
                    events.at[row_index, "pillar_method"] = "minilm_financial_fallback"
                    events.at[row_index, "financial_fallback_score"] = float(score)
                    events.at[row_index, "financial_fallback_accepted"] = 1
                else:
                    events.at[row_index, "pillar_label"] = "unclassified"
                    events.at[row_index, "pillar_method"] = "unclassified_after_financial_fallback"
                    events.at[row_index, "inference_status"] = "unclassified_no_financial_evidence"

        events["direction_input_text"] = ""
        for label in DIRECTION_LABELS:
            events[f"p_{label}"] = np.nan
        events["direction_label"] = pd.NA
        events["signed_tone"] = np.nan
        events["direction_confidence"] = np.nan

        score_indices = events.index[events["pillar_label"].isin(PILLARS)].tolist()
        if score_indices:
            direction_inputs = [
                build_direction_input(row.company_name, row.pillar_label, row.headline)
                for row in events.loc[score_indices].itertuples()
            ]
            events.loc[score_indices, "direction_input_text"] = direction_inputs
            direction_rows = _probability_rows(
                self.direction_classifier.predict_proba(direction_inputs),
                DIRECTION_LABELS,
                len(score_indices),
            )
            for row_index, probabilities in zip(score_indices, direction_rows):
                for label in DIRECTION_LABELS:
                    events.at[row_index, f"p_{label}"] = probabilities[label]
                events.at[row_index, "direction_label"] = max(
                    DIRECTION_LABELS, key=probabilities.get
                )
                events.at[row_index, "signed_tone"] = signed_tone(probabilities)
                events.at[row_index, "direction_confidence"] = direction_confidence(probabilities)
                events.at[row_index, "inference_status"] = "direction_scored"

        return events


def build_default_pipeline(
    direction_model_path: str | Path,
    *,
    direction_model_revision: str | None = None,
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> NewsPipeline:
    """Configure the three lazy model layers used by the prototype."""

    return NewsPipeline(
        esg_classifier=TransformersClassifier(
            DEFAULT_ESG_MODEL,
            ESG_LABELS,
            revision=DEFAULT_ESG_REVISION,
            device=device,
            cache_dir=cache_dir,
        ),
        financial_fallback=MiniLMFinancialFallback(
            device=device, cache_dir=cache_dir
        ),
        direction_classifier=TransformersClassifier(
            direction_model_path,
            DIRECTION_LABELS,
            revision=direction_model_revision,
            device=device,
            cache_dir=cache_dir,
        ),
    )


def _weighted_mean(frame: pd.DataFrame, column: str, weights: np.ndarray) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any() or weights[valid].sum() <= 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def aggregate_event_features(
    scored_events: pd.DataFrame,
    company_ids: Sequence[str],
    as_of_utc: str | pd.Timestamp,
    *,
    windows: Sequence[int] = (30, 90, 180),
    recency_half_life_days: float = 30.0,
    shrinkage_target: float = 3.0,
) -> pd.DataFrame:
    """Create one event-weighted row per company, pillar, and time window."""

    required = {
        "company_id",
        "event_id",
        "pillar_label",
        "published_at_utc",
        "signed_tone",
    }
    missing = sorted(required - set(scored_events.columns))
    if missing:
        raise ValueError(f"Missing required scored-event columns: {missing}")
    if scored_events.duplicated(["company_id", "event_id"]).any():
        raise ValueError("Each company_id and event_id pair must be unique.")
    if recency_half_life_days <= 0 or shrinkage_target <= 0:
        raise ValueError("Weighting parameters must be positive.")
    if not windows or any(int(window) <= 0 for window in windows):
        raise ValueError("windows must contain positive day counts.")

    companies = list(dict.fromkeys(str(company_id) for company_id in company_ids))
    prepared = scored_events.loc[
        scored_events["pillar_label"].isin(PILLARS) & scored_events["signed_tone"].notna()
    ].copy()
    prepared["published_at"] = pd.to_datetime(
        prepared["published_at_utc"], utc=True, errors="coerce"
    )
    if prepared["published_at"].isna().any():
        raise ValueError("A scored event has an invalid published_at_utc value.")
    as_of = _utc_timestamp(as_of_utc)
    prepared["age_days"] = (as_of - prepared["published_at"]).dt.total_seconds() / 86400
    prepared = prepared.loc[prepared["age_days"].ge(0)].copy()

    defaults: dict[str, object] = {
        "article_count": 1,
        "domains": "",
        "relevance_strength": 1.0,
        "direction_label": pd.NA,
        "p_negative": np.nan,
        "p_neutral": np.nan,
        "p_positive": np.nan,
        "esg_top_probability": np.nan,
        "financial_fallback_score": np.nan,
        "direction_confidence": np.nan,
    }
    for column, default in defaults.items():
        if column not in prepared:
            prepared[column] = default
    prepared["relevance_strength"] = pd.to_numeric(
        prepared["relevance_strength"], errors="raise"
    )
    if not prepared["relevance_strength"].between(0, 1).all():
        raise ValueError("relevance_strength must be within [0, 1].")

    rows: list[dict[str, object]] = []
    for window in (int(value) for value in windows):
        inside = prepared.loc[prepared["age_days"].le(window)].copy()
        inside["recency_weight"] = np.power(
            0.5, inside["age_days"] / recency_half_life_days
        )
        inside["event_weight"] = inside["relevance_strength"] * inside["recency_weight"]

        for company_id in companies:
            for pillar in PILLARS:
                group = inside.loc[
                    inside["company_id"].astype(str).eq(company_id)
                    & inside["pillar_label"].eq(pillar)
                ]
                base = {"company_id": company_id, "pillar": pillar, "window_days": window}
                if group.empty:
                    rows.append(
                        {
                            **base,
                            "event_count": 0,
                            "article_count": 0,
                            "source_count": 0,
                            "active_days": 0,
                            "effective_weight": 0.0,
                            "has_evidence": 0,
                            "tone_raw": np.nan,
                            "tone_shrunk_to_zero": 0.0,
                            "mean_p_negative": np.nan,
                            "mean_p_neutral": np.nan,
                            "mean_p_positive": np.nan,
                            "negative_event_share": np.nan,
                            "neutral_event_count": 0,
                            "mean_esg_top_probability": np.nan,
                            "mean_financial_fallback_score": np.nan,
                            "mean_relevance_strength": np.nan,
                            "mean_direction_confidence": np.nan,
                            "latest_event_age_days": np.nan,
                        }
                    )
                    continue

                weights = group["event_weight"].to_numpy(float)
                weight_sum = float(weights.sum())
                raw_tone = _weighted_mean(group, "signed_tone", weights)
                shrink = min(1.0, weight_sum / shrinkage_target)
                domains = {
                    domain for packed in group["domains"] for domain in _split_values(packed)
                }
                negative = group["direction_label"].eq("negative").to_numpy(float)
                negative_share = (
                    float(np.average(negative, weights=weights)) if weight_sum > 0 else np.nan
                )
                rows.append(
                    {
                        **base,
                        "event_count": int(group["event_id"].nunique()),
                        "article_count": int(
                            pd.to_numeric(group["article_count"], errors="coerce").fillna(1).sum()
                        ),
                        "source_count": len(domains),
                        "active_days": int(group["published_at"].dt.date.nunique()),
                        "effective_weight": weight_sum,
                        "has_evidence": 1,
                        "tone_raw": raw_tone,
                        "tone_shrunk_to_zero": raw_tone * shrink,
                        "mean_p_negative": _weighted_mean(group, "p_negative", weights),
                        "mean_p_neutral": _weighted_mean(group, "p_neutral", weights),
                        "mean_p_positive": _weighted_mean(group, "p_positive", weights),
                        "negative_event_share": negative_share,
                        "neutral_event_count": int(group["direction_label"].eq("neutral").sum()),
                        "mean_esg_top_probability": _weighted_mean(
                            group, "esg_top_probability", weights
                        ),
                        "mean_financial_fallback_score": _weighted_mean(
                            group, "financial_fallback_score", weights
                        ),
                        "mean_relevance_strength": _weighted_mean(
                            group, "relevance_strength", weights
                        ),
                        "mean_direction_confidence": _weighted_mean(
                            group, "direction_confidence", weights
                        ),
                        "latest_event_age_days": float(group["age_days"].min()),
                    }
                )
    return pd.DataFrame(rows)


def compute_tone_momentum(
    features: pd.DataFrame,
    *,
    short_window: int = 30,
    baseline_window: int = 90,
) -> pd.DataFrame:
    """Return short-window minus baseline-window shrunk tone by company and pillar."""

    required = {
        "company_id",
        "pillar",
        "window_days",
        "tone_shrunk_to_zero",
        "has_evidence",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if short_window == baseline_window:
        raise ValueError("Momentum windows must be different.")
    if features.duplicated(["company_id", "pillar", "window_days"]).any():
        raise ValueError("Feature rows must be unique by company, pillar, and window.")

    keys = ["company_id", "pillar"]
    value_columns = [*keys, "tone_shrunk_to_zero", "has_evidence"]
    short = features.loc[
        features["window_days"].eq(short_window), value_columns
    ].rename(
        columns={
            "tone_shrunk_to_zero": "short_tone",
            "has_evidence": "short_window_has_evidence",
        }
    )
    baseline = features.loc[
        features["window_days"].eq(baseline_window), value_columns
    ].rename(
        columns={
            "tone_shrunk_to_zero": "baseline_tone",
            "has_evidence": "baseline_window_has_evidence",
        }
    )
    result = short.merge(baseline, on=keys, how="inner", validate="one_to_one")
    expected_pairs = features[keys].drop_duplicates()
    if len(result) != len(expected_pairs):
        raise ValueError("Both requested windows are required for every company and pillar.")
    result["tone_momentum"] = result["short_tone"] - result["baseline_tone"]
    result["short_window_days"] = int(short_window)
    result["baseline_window_days"] = int(baseline_window)
    return result[
        [
            *keys,
            "short_window_days",
            "baseline_window_days",
            "tone_momentum",
            "short_window_has_evidence",
            "baseline_window_has_evidence",
        ]
    ].sort_values(keys).reset_index(drop=True)


__all__ = [
    "DIRECTION_LABELS",
    "ESG_LABELS",
    "PILLARS",
    "MiniLMFinancialFallback",
    "NewsPipeline",
    "TransformersClassifier",
    "aggregate_event_features",
    "build_default_pipeline",
    "build_direction_input",
    "compute_tone_momentum",
    "deduplicate_events",
    "direction_confidence",
    "issuer_aliases",
    "keyword_hits",
    "mask_company_aliases",
    "normalize_text",
    "signed_tone",
]
