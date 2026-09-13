# ETHack2026

Climate Transition Score notebook and outputs for the ETHack 2026 sustainability challenge.

## Challenge goal

The challenge asks us to build a data-driven framework to quantify and compare the sustainability of companies in the S&P 500. The methodology is intentionally open-ended: teams must define what sustainability means, choose and justify relevant indicators, source suitable data, and develop a transparent way to score, rank, or compare companies.

The bonus scenario asks how a $1 billion investment fund should be allocated if the world suddenly commits to reaching net-zero emissions as fast as possible. This repository focuses on the first required step for that scenario: measuring corporate climate transition readiness. It does not directly build the portfolio allocation yet.

## Project approach

This project interprets sustainability through a climate-transition lens rather than a broad ESG lens. The central question is:

> Which S&P 500 companies have the strongest evidence of low current emissions exposure, actual emissions reductions, and credible forward-looking transition commitments?

The notebook builds a transparent Climate Transition Score from three components:

1. **Current Footprint**: where the company is today, based on FY2024 emissions intensity.
2. **Realized Decarbonization**: whether the company has actually reduced emissions over 2020-2024.
3. **Forward Target / Alignment**: whether the company has credible SBTi target information.

Realized decarbonization receives the largest baseline weight because actual emissions reductions should matter more than corporate commitments alone.

## Methodology summary

The score keeps Scope 1+2 emissions separate from Scope 3 emissions because Scope 3 reporting boundaries are less comparable across companies. It also combines global and peer-relative percentiles:

- The global score asks how carbon-efficient a company is compared with the market.
- The peer score asks how well a company is transitioning compared with firms facing similar sector constraints.

Missing data are not filled with zero or sample averages. Instead, the notebook calculates data coverage and shrinks incomplete scores toward a neutral value, so limited disclosure is not automatically rewarded or punished.

## Repository contents

- `notebooks/climate_transition_score.ipynb`: executed notebook with data inspection, methodology, diagnostics, rankings, and charts.
- `data/ESGData.xlsx`: source workbook used by the notebook.
- `data/ESGData_Practical_Data_Dictionary.pdf`: source data dictionary, included for reference.
- `outputs/climate_transition_scores.csv`: exported company-level score dataset.
- `outputs/climate_transition_scores.xlsx`: Excel version of the exported score dataset.

## News NLP prototype

The news extension adds a transparent current-event signal without mixing news counts into the climate score. Start with the [pipeline picture and formulas](docs/news_nlp/news_inference_pipeline.md), review the [FinBERT direction fine-tuning](docs/news_nlp/finbert_direction_model.md), or run the [small synthetic demo](notebooks/news_pipeline_demo.ipynb). The reusable logic lives in `src/news_nlp.py`; large weights, private labels, and raw news caches stay outside Git.

Install the small project environment with `pip install -r requirements-news-nlp.txt`. The demo runs offline by default; real inference additionally requires the selected direction checkpoint through `DIRECTION_MODEL_PATH_OR_ID`.

## Portfolio extension

The exported Climate Transition Score can later be combined with a financial transition-risk or transition-beta signal to answer the bonus portfolio question. In that later step, the investment signal should distinguish:

- climate transition readiness
- financial exposure to a net-zero transition shock
- portfolio risk and diversification constraints

This repository currently provides the climate transition readiness input.
