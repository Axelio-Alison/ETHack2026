# Fine-tuning FinBERT for news direction

## What the model does

Generic financial sentiment is not always the same as the effect of an event on a specific company. We therefore fine-tuned `ProsusAI/finbert` to classify a headline as **negative**, **neutral**, or **positive** for the named company and pillar.

The input format is deliberately simple and is kept unchanged during inference:

```text
Company: {company}. Pillar: {pillar}. Headline: {headline}
```

This model estimates headline direction. It does not choose the pillar and it does not produce a final company ESG score. The full routing process is described in the [news inference pipeline](news_inference_pipeline.md).

## Data and training

The final dataset contains 6,705 adjudicated examples: 5,730 for training, 119 for development, and 856 for testing. The test set combines 743 preserved legacy examples with 113 new, uncertainty-selected challenge examples. We added 757 active-learning labels after two independent model-review passes and model adjudication. These are carefully checked **silver labels**, not human gold labels.

We kept exact headlines, near-duplicates, and connected story families in the same split. Examples connected to the locked test set were excluded from training. Test labels were never used to select active-learning examples or the final seed.

| Setting | Value |
|---|---:|
| Base model | `ProsusAI/finbert` at revision `4556d130...` |
| Maximum length | 128 tokens |
| Learning rate | $2\times10^{-5}$ |
| Train / evaluation batch | 32 / 64 |
| Maximum epochs | 5 |
| Selected seed and epoch | 17 and 2 |
| Hardware | NVIDIA T4, mixed precision |

The loss uses square-root inverse-frequency class weights,

$$
w_c=\sqrt{\frac{N}{K n_c}},
$$

where $N$ is the training-set size, $K=3$, and $n_c$ is the number of training examples in class $c$. This reduces the influence of the majority neutral class without making rare examples dominate training.

## Evaluation

Macro-F1 gives each of the three labels the same importance:

$$
F_{1,c}=\frac{2\,\mathrm{precision}_c\,\mathrm{recall}_c}
{\mathrm{precision}_c+\mathrm{recall}_c},
\qquad
F_{1,\mathrm{macro}}=\frac{1}{3}\sum_c F_{1,c}.
$$

Both rows below use the same 856-example combined test set.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Original FinBERT tone labels | 63.43% | 0.5762 |
| Fine-tuned direction model | **81.07%** | **0.7793** |

On the deliberately difficult 113-example challenge subset, the pre-update model scored 52.21% accuracy and 0.5045 macro-F1. The selected update reached 68.14% and 0.6721. Seed 17 was selected by the predeclared rule—highest macro-F1 on the untouched development set—not by test performance.

As a retention check, macro-F1 on the 743 preserved legacy rows moved from 0.7980 to 0.7898 ($\Delta=-0.0082$). A paired bootstrap interval of $[-0.0343, 0.0182]$ includes zero, so we describe this as a small, inconclusive retention cost; the clear gain is on the new challenge cases.

| Pillar | Test rows | Macro-F1 |
|---|---:|---:|
| Financial | 566 | 0.7809 |
| Environmental | 41 | 0.5417 |
| Social | 102 | 0.6967 |
| Governance | 147 | 0.7069 |

## Reproducibility and limits

- Selected checkpoint SHA-256: `fc467b9ff6639aa4a77ae39ed7b486ecea36a6fc462206c91133308c525c6827`.
- See the [compact metrics](../../outputs/news_nlp/direction_metrics.json) and [reproduction notebook](../../notebooks/finbert_direction_finetuning.ipynb).
- Model weights are kept outside Git because the archive is about 406 MB.
- Pillars in this evaluation are adjudicated labels. These scores are not end-to-end pipeline accuracy.
- Only two new active-learning examples were environmental, so we cannot claim a meaningful environmental improvement.
- The model reads headlines, not full articles. Its output is a directional signal, not a causal conclusion, investment recommendation, or audited sustainability rating.

Base model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert). The fine-tuning code and derived checkpoint should be used subject to the upstream model terms.
