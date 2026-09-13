# Structured ESG score

Workflow:

1. Clean Bloomberg data, map securities to 500 companies, and join the regulatory panel.
2. Convert observed features into 0–100 “good performance” percentiles using sector peers, with a global fallback for small groups.
3. Calculate materiality-weighted pillar scores. When coverage is below 70%, shrink the score toward 50:

   `adjusted score = 50 + min(1, coverage / 0.70) × (raw score − 50)`

4. Combine pillars:

   `structured score = 45% Environmental + 15% Transition + 20% Social + 20% Governance`

Inputs by pillar: Environmental—Scope 1+2 footprint; Transition—emissions-intensity trend, SBTi, and climate governance; Social—diversity, safety, policies, and employee stability; Governance—board independence, CEO separation, attendance, women executives, and sustainability oversight.

The final score deducts up to 15 points for recent, sector-relevant regulatory evidence. A separate net-zero score combines absolute and employee-adjusted emissions trends with SBTi and climate-governance signals.
