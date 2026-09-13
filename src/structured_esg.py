"""Reusable mechanics for the structured S&P 500 sustainability notebook.

The notebook keeps the analytical choices visible. This module contains the
repeatable plumbing: input resolution, cleaning, entity mapping, normalization,
block scoring, audit-table construction, plotting, and JSON conversion.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import math
import os
import re

import numpy as np
import pandas as pd


_BLOOMBERG_ERROR_RE = re.compile(
    r"^#(?:N/A|VALUE!|REF!|DIV/0!|NAME\?|NUM!|NULL!|SPILL!|CALC!)",
    re.IGNORECASE,
)


def resolve_input(project_root: Path, env_name: str, filename: str) -> Path:
    """Return the first available location for a required input file."""
    candidates = [
        Path(os.environ[env_name]).expanduser() if os.environ.get(env_name) else None,
        project_root / "data" / filename,
        project_root / filename,
        Path("/content") / filename,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Set {env_name} or place {filename} in data/, the project root, or /content"
    )


def sector_levels(sectors, default="Medium", overrides=None):
    """Build a complete sector materiality row from a default and exceptions."""
    result = {sector: default for sector in sectors}
    result.update(overrides or {})
    return result


def clean_bloomberg_value(value):
    """Convert Bloomberg error strings to missing values and preserve real zeros."""
    if isinstance(value, str) and _BLOOMBERG_ERROR_RE.match(value.strip()):
        return np.nan
    return value


def read_bloomberg_sheet(worksheet):
    """Read populated Bloomberg rows and return a compact cleaning summary."""
    header = [
        str(value).strip() if value is not None else ""
        for value in next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True))
    ]
    rows = []
    skipped_rows = 0
    error_cells = 0
    numeric_zeroes = 0

    for raw in worksheet.iter_rows(min_row=2, values_only=True):
        # Ignore empty rows and formula artifacts below the table.
        if not any(value is not None and str(value).strip() for value in raw):
            skipped_rows += 1
            continue

        cleaned = []
        for value in raw:
            error_cells += int(
                isinstance(value, str) and bool(_BLOOMBERG_ERROR_RE.match(value.strip()))
            )
            numeric_zeroes += int(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value == 0
            )
            cleaned.append(clean_bloomberg_value(value))

        security_id = cleaned[0]
        if not isinstance(security_id, str) or not security_id.strip().endswith(" Equity"):
            skipped_rows += 1
            continue
        rows.append(cleaned)

    frame = pd.DataFrame(rows, columns=header)
    diagnostics = {
        "populated_security_rows": len(frame),
        "skipped_empty_or_artifact_rows": skipped_rows,
        "converted_error_cells": error_cells,
        "preserved_numeric_zero_cells": numeric_zeroes,
    }
    return frame, diagnostics


def normalize_ticker(value):
    """Convert Bloomberg ticker formats to the panel's ticker format."""
    text = str(value).upper().strip()
    text = re.sub(r"\s+[A-Z]{2}\s+EQUITY$", "", text)
    text = re.sub(r"\s+EQUITY$", "", text)
    text = text.replace("/", " ").replace(".", " ")
    return re.sub(r"\s+", " ", text).strip()


def _values_equal(left, right):
    """Compare values while treating two missing values as equal."""
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12))
    return str(left).strip() == str(right).strip()


def file_sha256(path: Path) -> str:
    """Create a reproducible fingerprint for an input file."""
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_security_map(companies: pd.DataFrame) -> pd.DataFrame:
    """Expand pipe-separated share classes into one mapping row per security."""
    # Use named intermediate steps so the identity mapping is easy to inspect.
    security_map = companies[
        ["company_id", "primary_ticker", "constituent_tickers"]
    ].copy()
    security_map["normalized_ticker"] = security_map["constituent_tickers"].str.split(
        "|", regex=False
    )
    security_map = security_map.explode("normalized_ticker")
    security_map["normalized_ticker"] = security_map["normalized_ticker"].map(
        normalize_ticker
    )
    security_map = security_map.drop(columns="constituent_tickers")
    security_map = security_map.reset_index(drop=True)

    if not security_map["normalized_ticker"].is_unique:
        raise ValueError("A normalized ticker maps to more than one company.")
    return security_map


def map_and_collapse(frame, sheet_name, security_map):
    """Map securities to companies and retain one verified row per company."""
    working = frame.copy()
    working["normalized_ticker"] = working["ID"].map(normalize_ticker)
    working = working.merge(
        security_map, on="normalized_ticker", how="left", validate="one_to_one"
    )

    if working["company_id"].isna().any():
        missing_ids = working.loc[working["company_id"].isna(), "ID"].tolist()
        raise ValueError(f"Unmapped securities in {sheet_name}: {missing_ids}")

    # Duplicate share classes must agree before one row is retained.
    identity_columns = {
        "ID",
        "name()",
        "id_isin()",
        "gics_sector_name()",
        "normalized_ticker",
        "company_id",
        "primary_ticker",
    }
    value_columns = [column for column in working if column not in identity_columns]
    conflicts = []
    for company_id, group in working.groupby("company_id", sort=False):
        if len(group) <= 1:
            continue
        for column in value_columns:
            values = group[column].tolist()
            if any(not _values_equal(values[0], value) for value in values[1:]):
                conflicts.append(
                    {
                        "sheet": sheet_name,
                        "company_id": company_id,
                        "field": column,
                        "security_values": {
                            row.ID: getattr(row, column)
                            for row in group.itertuples(index=False)
                        },
                    }
                )

    working["is_primary_security"] = working["normalized_ticker"].eq(
        working["primary_ticker"].map(normalize_ticker)
    )
    collapsed = working.sort_values(
        ["company_id", "is_primary_security"], ascending=[True, False]
    )
    collapsed = collapsed.drop_duplicates("company_id", keep="first")
    collapsed = collapsed.sort_values("company_id")
    collapsed = collapsed.reset_index(drop=True)
    return working, collapsed, conflicts


def select_features(collapsed, columns, prefix=None):
    """Select company-level fields and optionally prefix their names."""
    result = collapsed[["company_id", *columns]].copy()
    if prefix:
        result = result.rename(columns={column: f"{prefix}{column}" for column in columns})
    return result


def sum_if_complete(left, right):
    """Add two reported values only when both are available."""
    return left + right if pd.notna(left) and pd.notna(right) else np.nan


def annualized_log_trend(years, values, min_observations=4):
    """Estimate annual percentage change from positive, comparable observations."""
    pairs = [
        (year, value)
        for year, value in zip(years, values)
        if pd.notna(value) and float(value) > 0
    ]
    if len(pairs) < min_observations:
        return np.nan, len(pairs)
    x = np.array([year for year, _ in pairs], dtype=float)
    y = np.log(np.array([value for _, value in pairs], dtype=float))
    slope = np.polyfit(x, y, 1)[0]
    return float(np.expm1(slope) * 100), len(pairs)


def transition_classification(absolute_trend, intensity_trend):
    """Describe the direction of absolute and employee-adjusted emissions."""
    if pd.isna(absolute_trend) or pd.isna(intensity_trend):
        return "insufficient comparable observations"
    absolute = "absolute ↓" if absolute_trend <= 0 else "absolute ↑"
    intensity = "intensity ↓" if intensity_trend <= 0 else "intensity ↑"
    interpretation = {
        ("absolute ↓", "intensity ↓"): "strong transition",
        ("absolute ↑", "intensity ↓"): "efficiency improvement but growing footprint",
        ("absolute ↓", "intensity ↑"): "falling footprint with deteriorating efficiency",
        ("absolute ↑", "intensity ↑"): "deterioration",
    }[(absolute, intensity)]
    return f"{absolute} and {intensity}: {interpretation}"


def yn_numeric(series, good_value="Y"):
    """Map a Y/N field to 1/0 while leaving blanks missing."""
    mapping = {good_value: 1.0, ("N" if good_value == "Y" else "Y"): 0.0}
    return series.astype("string").str.upper().map(mapping).astype(float)


def percentile_good_scores(frame, value_column, beneficial, min_peers=15):
    """Return 0-100 scores and the peer group used for each observation."""
    values = pd.to_numeric(frame[value_column], errors="coerce")
    global_score = values.rank(method="average", pct=True) * 100
    if not beneficial:
        global_score = 100 - global_score

    sector_score = pd.Series(np.nan, index=frame.index, dtype=float)
    peer_group = pd.Series(pd.NA, index=frame.index, dtype="string")
    peer_count = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    fallback = pd.Series(False, index=frame.index, dtype=bool)
    global_n = int(values.notna().sum())

    for sector, idx in frame.groupby("sector").groups.items():
        observed_idx = [i for i in idx if pd.notna(values.loc[i])]
        sector_n = len(observed_idx)
        if sector_n >= min_peers:
            ranked = values.loc[observed_idx].rank(method="average", pct=True) * 100
            if not beneficial:
                ranked = 100 - ranked
            sector_score.loc[observed_idx] = ranked
            peer_group.loc[observed_idx] = f"sector:{sector}"
            peer_count.loc[observed_idx] = sector_n
        else:
            sector_score.loc[observed_idx] = global_score.loc[observed_idx]
            peer_group.loc[observed_idx] = "global fallback"
            peer_count.loc[observed_idx] = global_n
            fallback.loc[observed_idx] = True

    return pd.DataFrame(
        {
            "sector_score": sector_score.clip(0, 100),
            "global_score": global_score.clip(0, 100),
            "peer_group": peer_group,
            "peer_count": peer_count,
            "fallback_used": fallback,
        }
    )


def coverage_grade(weight, thresholds):
    """Translate observed evidence weight into a coverage grade."""
    if weight >= thresholds["High"]:
        return "High"
    if weight >= thresholds["Medium"]:
        return "Medium"
    return "Low"


def score_block(
    frame,
    feature_weights,
    materiality,
    threshold=0.70,
    neutral_prior=50.0,
    confidence_thresholds=None,
    prior_mode="fixed",
):
    """Score one block; the same function also handles sensitivity priors."""
    raw_scores = []
    observed_weights = []
    applicable_weights = []

    # Work company by company. This is more explicit than a vectorized expression
    # and keeps the treatment of missing and non-material features visible.
    for row in frame.itertuples(index=False):
        weighted_sum = 0.0
        observed_total = 0.0
        applicable_total = 0.0
        for feature, configured_weight in feature_weights.items():
            multiplier = materiality.get(feature, {}).get(row.sector, 1.0)
            applicable_weight = configured_weight * multiplier
            if applicable_weight <= 0:
                continue
            applicable_total += applicable_weight
            score = getattr(row, f"{feature}_score")
            if pd.notna(score):
                weighted_sum += applicable_weight * float(score)
                observed_total += applicable_weight

        raw_scores.append(weighted_sum / observed_total if observed_total else np.nan)
        observed_weights.append(
            observed_total / applicable_total if applicable_total else 0.0
        )
        applicable_weights.append(applicable_total)

    raw = pd.Series(raw_scores, index=frame.index, dtype=float)
    observed = pd.Series(observed_weights, index=frame.index, dtype=float).clip(0, 1)
    # Sensitivity tests reuse this function and change only the prior.
    if prior_mode == "sector_median":
        prior = raw.groupby(frame["sector"]).transform("median").fillna(neutral_prior)
    else:
        prior = pd.Series(float(neutral_prior), index=frame.index)

    # Sparse blocks move only part of the way from the prior to the raw score.
    evidence_factor = np.minimum(1.0, observed / threshold)
    adjusted = prior + evidence_factor * (raw.fillna(prior) - prior)
    thresholds = confidence_thresholds or {"High": 0.80, "Medium": 0.50}

    return pd.DataFrame(
        {
            "raw": raw,
            "adjusted": adjusted.clip(0, 100),
            "observed_weight": observed,
            "applicable_weight": applicable_weights,
            "confidence": observed.map(lambda value: coverage_grade(value, thresholds)),
        }
    )


def recency_weighted_metric(group, metric, match_flag, additional_eligibility=None):
    """Summarize one company's observed regulatory values across recent years."""
    eligible = group[match_flag].eq(1) & group[metric].notna()
    if additional_eligibility is not None:
        eligible &= additional_eligibility(group)
    observed = group.loc[eligible]
    if observed.empty:
        return pd.Series(
            {"value": np.nan, "years_observed": 0, "latest_observed_year": np.nan}
        )
    value = np.average(
        observed[metric].astype(float), weights=observed["recency_weight"].astype(float)
    )
    return pd.Series(
        {
            "value": float(value),
            "years_observed": len(observed),
            "latest_observed_year": int(observed["observation_year"].max()),
        }
    )


def regulatory_penalty(frame, source_weights, materiality, cap):
    """Combine observed, sector-applicable regulatory evidence into a deduction."""
    penalties = []
    observed_weights = []
    statuses = []
    for row in frame.itertuples(index=False):
        applicable_total = 0.0
        observed_total = 0.0
        adverse_sum = 0.0
        observed_sources = []

        for source, configured_weight in source_weights.items():
            source_weight = configured_weight * materiality[source].get(row.sector, 0.0)
            if source_weight <= 0:
                continue
            applicable_total += source_weight
            adverse_score = getattr(row, f"{source}_regulatory_adverse_score")
            if pd.notna(adverse_score):
                observed_total += source_weight
                adverse_sum += source_weight * float(adverse_score)
                observed_sources.append(source)

        observed_fraction = observed_total / applicable_total if applicable_total else 0.0
        if observed_total:
            composite_adverse = adverse_sum / observed_total
            penalty = min(cap, cap * composite_adverse / 100)
            status = "Observed applicable evidence: " + ", ".join(observed_sources)
        else:
            penalty = 0.0
            status = (
                "No observed applicable regulatory evidence; zero deduction is not a "
                "good-performance signal"
            )
        penalties.append(penalty)
        observed_weights.append(observed_fraction)
        statuses.append(status)

    return pd.DataFrame(
        {
            "penalty": penalties,
            "observed_weight": observed_weights,
            "status": statuses,
        },
        index=frame.index,
    )


def audit_feature_catalog(
    frame,
    catalog,
    audit_series,
    potential_conditions,
    original_weights,
    final_weights,
    sparse_threshold=0.10,
    redundancy_threshold=0.90,
):
    """Build the feature dictionary, correlations, and sector coverage tables."""
    audit_frame = pd.DataFrame(audit_series)
    correlations = audit_frame.corr(method="spearman", min_periods=20)
    redundant_pairs = []
    for position, left in enumerate(correlations.columns):
        for right in correlations.columns[position + 1 :]:
            rho = correlations.loc[left, right]
            if pd.notna(rho) and abs(rho) >= redundancy_threshold:
                redundant_pairs.append(
                    {"feature_1": left, "feature_2": right, "spearman": float(rho)}
                )

    dictionary_rows = []
    for spec in catalog:
        feature = spec["feature"]
        series = audit_series.get(feature)
        if series is not None:
            coverage = float(series.notna().mean())
            variance = (
                float(pd.to_numeric(series, errors="coerce").var())
                if series.notna().sum() > 1
                else np.nan
            )
        elif feature in potential_conditions:
            coverage = float(potential_conditions[feature].mean())
            variance = np.nan
        else:
            coverage = np.nan
            variance = np.nan

        flags = []
        if pd.notna(coverage) and coverage < sparse_threshold:
            flags.append("very sparse")
        if pd.notna(variance) and np.isclose(variance, 0):
            flags.append("zero variance")
        if any(feature in pair.values() for pair in redundant_pairs):
            flags.append("highly redundant")
        dictionary_rows.append(
            {
                "pillar": spec["pillar"],
                "output_feature": feature,
                "source_field": spec["source_field"],
                "description": spec["description"],
                "unit": spec["unit"],
                "direction": spec["direction"],
                "time_basis": spec["time_basis"],
                "numerator": spec["numerator"],
                "denominator": spec["denominator"],
                "missing_value_rule": spec["missing_rule"],
                "applicability_rule": spec["applicability_rule"],
                "overall_coverage": coverage,
                "variance": variance,
                "audit_flags": "; ".join(flags) if flags else "none",
                "feature_status": spec["status"],
                "decision_reason": spec["reason"],
                "original_weight": original_weights.get(spec["pillar"], {}).get(
                    feature, 0.0
                ),
                "final_weight": final_weights.get(spec["pillar"], {}).get(
                    feature, 0.0
                ),
                "economic_or_environmental_interpretation": spec["description"],
            }
        )

    coverage_rows = []
    for feature, series in {**audit_series, **potential_conditions}.items():
        observed = series.astype(bool) if pd.api.types.is_bool_dtype(series) else series.notna()
        for sector, idx in frame.groupby("sector").groups.items():
            coverage_rows.append(
                {
                    "feature": feature,
                    "sector": sector,
                    "coverage": float(observed.loc[idx].mean()),
                    "company_count": len(idx),
                }
            )

    return (
        pd.DataFrame(dictionary_rows),
        correlations,
        redundant_pairs,
        pd.DataFrame(coverage_rows),
    )


def feature_catalog(current_time_basis):
    """Return the auditable feature definitions used by the notebook."""
    shared = {
        "missing_rule": "Missing remains missing",
        "applicability_rule": "Sector materiality matrix",
    }

    def item(
        pillar, feature, source, description, unit, direction, time_basis,
        numerator, denominator, status, reason, **overrides,
    ):
        row = {
            "pillar": pillar,
            "feature": feature,
            "source_field": source,
            "description": description,
            "unit": unit,
            "direction": direction,
            "time_basis": time_basis,
            "numerator": numerator,
            "denominator": denominator,
            "status": status,
            "reason": reason,
            **shared,
        }
        row.update(overrides)
        return row

    # Each record reads in the same order as the helper signature above.
    return [
        item(
            "Environmental", "scope12_revenue_intensity",
            "GHG_SCOPE_1_2024 + GHG_SCOPE_2_2024 / SALES_REV_TURN_2024",
            "Scope 1+2 emissions per revenue",
            "Not calculated: emissions physical unit and sales reporting currency absent",
            "adverse", "2024", "GHG_SCOPE_1_2024 + GHG_SCOPE_2_2024", "SALES_REV_TURN_2024",
            "Disabled", "Failed unit/currency comparability check",
        ),
        item(
            "Environmental", "scope3_revenue_intensity",
            "GHG_SCOPE_3_2024 / SALES_REV_TURN_2024", "Scope 3 emissions per revenue",
            "Not calculated: emissions physical unit and sales reporting currency absent",
            "adverse", "2024", "GHG_SCOPE_3_2024", "SALES_REV_TURN_2024",
            "Disabled", "Failed unit/currency comparability check; lower coverage",
        ),
        item(
            "Environmental", "energy_revenue_intensity",
            "ENERGY_CONSUMPTION / SALES_REV_TURN_2024", "Energy use per revenue",
            "Not calculated: energy unit, aligned period, and sales reporting currency absent",
            "adverse", "Mixed candidate; invalid", "ENERGY_CONSUMPTION", "SALES_REV_TURN_2024",
            "Disabled", "Failed unit, period, and currency checks",
        ),
        item(
            "Environmental", "resource_revenue_intensity",
            "WATER_CONSUMPTION; TOTAL_WASTE / SALES_REV_TURN_2024", "Resource use per revenue",
            "Not calculated: incompatible resource units and sales currency absent",
            "adverse", "Mixed candidate; invalid", "Water/waste candidate fields", "SALES_REV_TURN_2024",
            "Disabled", "No coherent verified numerator; failed currency check",
        ),
        item(
            "Environmental", "renewable_energy_ratio",
            "RENEW_ENERGY_USE / ENERGY_CONSUMPTION", "Renewable share of energy use",
            "Not calculated: exact source units and aligned period absent",
            "beneficial", current_time_basis, "RENEW_ENERGY_USE", "ENERGY_CONSUMPTION",
            "Disabled", "Exact numerator/denominator units and period not documented",
        ),
        item(
            "Environmental", "scope12_reported_amount_2024",
            "GHG_SCOPE_1_2024 + GHG_SCOPE_2_2024",
            "Reported Scope 1+2 footprint, used only as an ordinal outcome",
            "Bloomberg native reported amount; exact physical unit not supplied",
            "adverse", "2024", "GHG_SCOPE_1_2024 + GHG_SCOPE_2_2024", "None",
            "Enabled", "Same standardized fields support ordinal sector/global percentiles; no physical-unit claim",
            missing_rule="Missing if either scope is missing",
        ),
        item(
            "Transition", "scope12_employee_intensity_trend",
            "GHG_SCOPE_1/2_2019:2024 and NUM_OF_EMPLOYEES_2019:2024",
            "Annualized trend in reported Scope 1+2 amount per employee",
            "percent per year; unknown emissions scale cancels",
            "adverse", "2019-2024; >=4 comparable years",
            "Annual Scope 1+2 reported amount", "Annual employee count", "Enabled",
            "Within-company log trend is scale-invariant; employee denominator is a count",
            missing_rule="Missing with fewer than four positive comparable pairs",
        ),
        item(
            "Transition", "scope12_absolute_trend", "GHG_SCOPE_1/2_2019:2024",
            "Annualized trend in absolute reported Scope 1+2 amount",
            "percent per year; unknown emissions scale cancels",
            "adverse", "2019-2024; >=4 comparable years",
            "Annual Scope 1+2 reported amount", "None", "Enabled",
            "Within-company log trend is scale-invariant",
            missing_rule="Missing with fewer than four positive comparable observations",
        ),
        item(
            "Transition", "sbti_status", "SBTI_NEAR_TERM_TARGET_STATUS",
            "SBTi near-term target status", "ordinal category", "beneficial", current_time_basis,
            "Not applicable", "Not applicable", "Enabled", "Verified categorical commitment field",
            missing_rule="Missing is not a failed commitment",
        ),
        item(
            "Transition", "climate_governance_support", "CSR_SUSTAINABILITY_COMMITTEE",
            "Sustainability committee as climate-governance support proxy", "Y/N", "beneficial",
            current_time_basis, "Not applicable", "Not applicable", "Enabled",
            "Broad governance proxy; not treated as realized climate performance",
            missing_rule="Missing is unscored",
        ),
        item(
            "Social", "diversity", "PCT_WOMEN_EMPLOYEES; PCT_WOMEN_MGT",
            "Average observed workforce and management gender representation",
            "percent", "beneficial", current_time_basis,
            "Reported percentages", "Reported workforce populations", "Enabled",
            "Direct representation measures with explicit percent units",
            missing_rule="Average observed subfields only",
        ),
        item(
            "Social", "employee_safety", "WORK_ACCIDENTS_EMPLOYEES; FATALITIES_EMPLOYEES",
            "Employee safety evidence from reported event counts", "reported counts", "adverse",
            current_time_basis, "Reported employee events", "None aligned; therefore downweighted",
            "Enabled - downweighted", "Useful evidence but no aligned exposure denominator",
            missing_rule="Average observed percentile subfields only",
        ),
        item(
            "Social", "social_policy", "Five Y/N policy fields", "Observed social-policy composite",
            "share of observed Y/N fields", "beneficial", current_time_basis,
            "Positive observed policies", "Observed policy fields only", "Enabled",
            "Transparent policy breadth measure",
            missing_rule="Missing fields excluded; no all-missing score",
        ),
        item(
            "Social", "employee_stability", "EMPLOYEE_TURNOVER_PCT", "Employee turnover",
            "percent", "adverse", current_time_basis, "Reported leavers", "Reported workforce basis",
            "Enabled", "Explicit percentage; coverage is reported", missing_rule="Missing is unscored",
        ),
        item(
            "Governance", "board_independence", "PCT_INDEPENDENT_DIRECTORS",
            "Independent directors", "percent", "beneficial", current_time_basis,
            "Independent directors", "Board members", "Enabled", "Direct governance outcome",
            missing_rule="Missing is unscored", applicability_rule="Universal",
        ),
        item(
            "Governance", "ceo_separation", "CEO_DUALITY", "CEO and chair roles separated",
            "Y/N transformed so N duality is beneficial", "beneficial", current_time_basis,
            "Not applicable", "Not applicable", "Enabled", "Direct governance structure",
            missing_rule="Missing is unscored", applicability_rule="Universal",
        ),
        item(
            "Governance", "board_attendance", "BOARD_MEETING_ATTENDANCE_PCT",
            "Board meeting attendance", "percent", "beneficial", current_time_basis,
            "Attended meetings", "Applicable board meetings", "Enabled", "Direct board-function measure",
            missing_rule="Missing is unscored", applicability_rule="Universal",
        ),
        item(
            "Governance", "women_executives", "PCT_OF_EXECUTIVES_THAT_ARE_WOMEN",
            "Women among executives", "percent", "beneficial", current_time_basis,
            "Women executives", "Executives", "Enabled", "Direct leadership-diversity measure",
            missing_rule="Missing is unscored", applicability_rule="Universal",
        ),
        item(
            "Governance", "sustainability_committee", "CSR_SUSTAINABILITY_COMMITTEE",
            "Board/company sustainability committee", "Y/N", "beneficial", current_time_basis,
            "Not applicable", "Not applicable", "Enabled",
            "Oversight structure; not a realized outcome",
            missing_rule="Missing is unscored", applicability_rule="Universal",
        ),
    ]


def build_inspection_tables(frame, scores, pillar_keys, largest_penalties):
    """Build compact top/bottom and manual-review tables."""
    eligible = scores.loc[~scores["overall_coverage_grade"].eq("Low")].copy()
    top = eligible.nlargest(5, "structured_score_after_regulatory_penalty").copy()
    bottom = eligible.nsmallest(5, "structured_score_after_regulatory_penalty").copy()
    top["table_position"] = "Top"
    bottom["table_position"] = "Bottom"
    top_bottom = pd.concat([top, bottom], ignore_index=True)

    def explanation(row):
        components = {
            "E": row["environmental_adjusted_score"],
            "T": row["transition_adjusted_score"],
            "S": row["social_adjusted_score"],
            "G": row["governance_adjusted_score"],
        }
        high = max(components, key=components.get)
        low = min(components, key=components.get)
        return (
            f"Strongest {high} {components[high]:.1f}; weakest {low} "
            f"{components[low]:.1f}; penalty "
            f"{row['regulatory_evidence_penalty_provisional']:.1f}."
        )

    top_bottom["component_explanation"] = top_bottom.apply(explanation, axis=1)
    top_bottom["confidence_warning"] = np.where(
        top_bottom["overall_coverage_grade"].eq("Low"),
        "LOW CONFIDENCE - diagnostic only",
        "",
    )

    parts = []
    for prefix, title in pillar_keys.items():
        columns = [
            "company_id",
            "primary_ticker",
            "company_name",
            "sector",
            f"{prefix}_raw",
            f"{prefix}_adjusted",
            f"{prefix}_observed_weight",
            f"{prefix}_confidence",
        ]
        high = frame.nlargest(5, f"{prefix}_adjusted")[columns].copy()
        low = frame.nsmallest(5, f"{prefix}_adjusted")[columns].copy()
        high["inspection_type"], high["block"] = "highest", title
        low["inspection_type"], low["block"] = "lowest", title
        parts.extend([high, low])

    gaps = frame.reindex(
        frame["credibility_gap"].abs().sort_values(ascending=False).index
    ).head(10)[
        [
            "company_id",
            "primary_ticker",
            "company_name",
            "sector",
            "commitment_score",
            "realized_transition_score",
            "credibility_gap",
            "commitment_observed_weight",
            "realized_transition_observed_weight",
        ]
    ].copy()
    gaps["inspection_type"], gaps["block"] = (
        "largest absolute gap",
        "Credibility gap",
    )
    penalties = largest_penalties.copy()
    penalties["inspection_type"], penalties["block"] = (
        "largest penalty",
        "Regulatory penalty",
    )
    manual = pd.concat(parts + [gaps, penalties], ignore_index=True, sort=False)
    penalty_present = manual.get(
        "regulatory_evidence_penalty_provisional", pd.Series(index=manual.index)
    ).notna()
    manual["inspection_note"] = np.where(
        penalty_present,
        "Review the source-specific evidence used for this penalty.",
        "Interpret the extreme together with observed weight; shrinkage limits "
        "sparse-evidence extremes.",
    )
    return top_bottom, manual


def create_figures(frame, output_dir, sectors, pillar_weights, largest_rank_changes, top_bottom):
    """Create the six diagnostic figures and return their supporting data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    output_dir = Path(output_dir)

    # Figure 1: decompose one representative company's score.
    # Choose a company near the median score and median coverage.
    eligible = frame.loc[~frame["overall_coverage_grade"].eq("Low")].copy()
    score_distance = (
        eligible["structured_score_after_regulatory_penalty"]
        - eligible["structured_score_after_regulatory_penalty"].median()
    ).abs()
    coverage_distance = (
        eligible["overall_observed_weight"] - eligible["overall_observed_weight"].median()
    ).abs()
    distance = score_distance + 10 * coverage_distance
    representative = frame.loc[distance.idxmin()]

    labels = ["Environmental", "Transition", "Social", "Governance"]
    values = [representative[f"{label.lower()}_adjusted"] for label in labels]
    weights = [pillar_weights[label] for label in labels]
    contributions = [value * weight for value, weight in zip(values, weights)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#2A9D8F", "#457B9D", "#E9C46A", "#6D597A"]
    bars = ax.barh(labels, contributions, color=colors)
    for bar, value, weight in zip(bars, values, weights):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} x {weight:.0%}",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Weighted contribution to structured score")
    title = f"Score decomposition: {representative['company_name']} ({representative['primary_ticker']})"
    caption = (
        f"Before penalty {representative['structured_score_before_regulatory_penalty']:.1f}; "
        f"provisional penalty {representative['regulatory_evidence_penalty_provisional']:.1f}; "
        f"after penalty {representative['structured_score_after_regulatory_penalty']:.1f}; "
        f"coverage {representative['overall_observed_weight']:.0%} "
        f"({representative['overall_coverage_grade']})."
    )
    ax.set_title(title)
    ax.text(0.01, -0.20, caption, transform=ax.transAxes, fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "01_score_decomposition_representative_company.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: show which components are observed in each sector.
    component_columns = {
        "E: Scope 1+2 amount": "scope12_reported_amount_2024_score",
        "T: Intensity trend": "scope12_employee_intensity_trend_score",
        "T: SBTi": "sbti_status_score",
        "T: Climate oversight": "climate_governance_support_score",
        "S: Diversity": "diversity_score",
        "S: Safety": "employee_safety_score",
        "S: Policies": "social_policy_score",
        "S: Stability": "employee_stability_score",
        "G: Independence": "board_independence_score",
        "G: CEO separation": "ceo_separation_score",
        "G: Attendance": "board_attendance_score",
        "G: Women executives": "women_executives_score",
        "G: Sustainability committee": "sustainability_committee_score",
        "Reg: TRI": "tri_regulatory_adverse_score",
        "Reg: CFPB": "cfpb_regulatory_adverse_score",
        "Reg: CPSC": "cpsc_regulatory_adverse_score",
        "Reg: openFDA": "openfda_regulatory_adverse_score",
    }
    coverage = pd.DataFrame(index=sectors)
    for label, column in component_columns.items():
        observed = frame.assign(observed=frame[column].notna())
        sector_coverage = observed.groupby("sector")["observed"].mean()
        coverage[label] = sector_coverage.reindex(sectors)
    fig, ax = plt.subplots(figsize=(17, 7))
    sns.heatmap(
        coverage * 100,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Companies with observed component (%)"},
        ax=ax,
    )
    ax.set(title="Coverage by sector and component", xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(output_dir / "02_coverage_heatmap_by_sector_component.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: compare the broad score with the narrower transition score.
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {"High": "#2A9D8F", "Medium": "#E9C46A", "Low": "#E76F51"}
    for grade, group in frame.groupby("overall_coverage_grade"):
        ax.scatter(
            group["structured_score_after_regulatory_penalty"],
            group["net_zero_transition_score"],
            s=38,
            alpha=0.75,
            label=f"{grade} coverage",
            color=palette[grade],
        )
    gaps = frame.copy()
    gaps["signed_gap"] = (
        gaps["structured_score_after_regulatory_penalty"] - gaps["net_zero_transition_score"]
    )
    highlights = pd.concat([
        gaps.nlargest(2, "signed_gap"),
        gaps.nsmallest(2, "signed_gap"),
    ]).drop_duplicates("company_id")
    annotation_offsets = [(12, 24), (12, -24), (-12, 24), (-12, -24)]
    for row, (dx, dy) in zip(highlights.itertuples(index=False), annotation_offsets):
        ax.annotate(
            row.primary_ticker,
            (row.structured_score_after_regulatory_penalty, row.net_zero_transition_score),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha="left" if dx > 0 else "right",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
        )
    ax.set(
        xlabel="Structured sustainability score after provisional penalty",
        ylabel="Net-zero transition score",
        title="Broad sustainability and net-zero transition are distinct",
    )
    ax.legend(frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "03_sustainability_vs_net_zero_transition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: isolate the largest rank movements caused by the deduction.
    rank_shift = largest_rank_changes.sort_values("rank_delta")
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_colors = np.where(rank_shift["rank_delta"] < 0, "#D1495B", "#2A9D8F")
    ax.barh(rank_shift["primary_ticker"], rank_shift["rank_delta"], color=bar_colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set(
        xlabel="Rank delta (negative = deterioration after regulatory evidence)",
        ylabel="",
        title="Largest rank changes from provisional regulatory evidence",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "04_regulatory_rank_shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: contrast stated commitments with realized transition evidence.
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(
        frame["realized_transition_score"],
        frame["commitment_score"],
        c=frame["credibility_gap"],
        cmap="RdYlGn_r",
        vmin=-40,
        vmax=40,
        s=42,
        alpha=0.8,
    )
    ax.plot([0, 100], [0, 100], linestyle="--", color="#555555", linewidth=1)
    large_gaps = frame.nlargest(8, "credibility_gap")
    ax.scatter(
        large_gaps["realized_transition_score"],
        large_gaps["commitment_score"],
        s=95,
        facecolors="none",
        edgecolors="#222222",
        linewidths=1.0,
    )
    gap_offsets = [(28, 32), (42, 0), (28, -32)]
    for row, (dx, dy) in zip(large_gaps.head(3).itertuples(index=False), gap_offsets):
        ax.annotate(
            row.primary_ticker,
            (row.realized_transition_score, row.commitment_score),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
        )
    ax.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xlabel="Realized transition score",
        ylabel="Commitment score",
        title="Commitments versus realized outcomes",
    )
    fig.colorbar(points, ax=ax, label="Credibility gap (commitment - realized)")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "05_commitments_vs_realized_outcomes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: render the compact top-and-bottom table.
    table_columns = [
        "table_position",
        "primary_ticker",
        "company_name",
        "structured_score_after_regulatory_penalty",
        "net_zero_transition_score",
        "overall_coverage_grade",
    ]
    table_view = top_bottom[table_columns].copy()
    table_view["structured_score_after_regulatory_penalty"] = table_view[
        "structured_score_after_regulatory_penalty"
    ].map(lambda value: f"{value:.1f}")
    table_view["net_zero_transition_score"] = table_view["net_zero_transition_score"].map(
        lambda value: f"{value:.1f}"
    )
    table_view.columns = ["Group", "Ticker", "Company", "Structured", "Net-zero", "Coverage"]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.axis("off")
    table = ax.table(
        cellText=table_view.values,
        colLabels=table_view.columns,
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=[0.08, 0.08, 0.36, 0.11, 0.11, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif table_view.iloc[row - 1, 0] == "Top":
            cell.set_facecolor("#E8F4F1")
        else:
            cell.set_facecolor("#FCECEE")
    ax.set_title(
        "Top and bottom structured scores among Medium/High coverage companies\n"
        "Component explanations are saved in the accompanying CSV",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "06_top_bottom_structured_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return representative, coverage


def json_safe(value):
    """Convert pandas and NumPy values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value
