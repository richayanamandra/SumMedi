# BioBERT Discriminator — Factual Consistency Classifier

This component implements the **discriminator** side of the SumMedi GAN architecture. A BioBERT-based binary classifier is fine-tuned to evaluate the factual consistency between a source medical document and a generated summary, producing a faithfulness score used as an adversarial signal during BART training.

---

## Overview

Hallucination is a critical problem in medical summarization — models like BART can generate fluent but factually incorrect summaries. The discriminator addresses this by acting as a learned factual verifier: given a (document, summary) pair, it predicts whether the summary is **faithful** (label = 1) or **hallucinated** (label = 0).

During adversarial training, this score is fed back to the BART generator as a reward signal, penalizing hallucinated outputs and encouraging factually grounded summaries.

---

## Repository Structure

```
biobert_discriminator/
├── BioBERT_Discriminator_MS2.ipynb   # Full training pipeline
├── biobert_discriminator_best/       # Saved model checkpoint (weights shared separately)
│   ├── config.json
│   ├── tokenizer_config.json
│   └── pytorch_model.bin
└── README.md                         # This file
```

---

## Model

- **Base model:** `dmis-lab/biobert-base-cased-v1.2`
- **Task:** Binary sequence classification — Faithful (1) vs Hallucinated (0)
- **Input format:** `[CLS] document [SEP] summary [SEP]`
- **Output:** Faithfulness probability ∈ [0, 1]

BioBERT was chosen over standard BERT because it is pre-trained on biomedical corpora (PubMed abstracts and PMC full-text articles), giving it strong domain-specific language understanding out of the box.

---

## Dataset

**MS2 — Multi-Document Summarization of Medical Studies**  
Accessed via: `allenai/mslr2022` (subset: `ms2`) on Hugging Face

| Split | Samples used |
|-------|-------------|
| Train | 2,000 source reviews → 4,000 examples (1:1 positive/negative) |
| Validation | 400 source reviews → 800 examples |

Since MS2 only contains faithful reference summaries, negative (hallucinated) examples are simulated during this initial phase using three perturbation strategies:

| Strategy | Description |
|----------|-------------|
| `random_replace` | Replace summary with one from a different document |
| `sentence_shuffle` | Randomly reorder sentences within the summary |
| `word_swap` | Replace ~20% of words with words from another summary |

> **Note:** These are placeholder negatives for the standalone discriminator phase. Once BART is fine-tuned and generating summaries, the discriminator will be retrained using real BART-generated outputs as negatives — which more closely resemble actual model hallucinations.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 4 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 10% |
| Max sequence length | 512 |
| Optimizer | AdamW (weight decay 0.01) |
| Gradient clipping | 1.0 |

---

## Inference

The `score_summary()` function is the interface that plugs into the adversarial training loop:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F

def load_discriminator(path='biobert_discriminator_best'):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.eval()
    return tok, mdl

def score_summary(document, summary, tok, mdl, max_len=512):
    enc = tok(document, summary, max_length=max_len,
              padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        probs = F.softmax(mdl(**enc).logits, dim=1)
    return probs[0][1].item()  # faithfulness probability
```

---

## Integration with BART Generator

This discriminator is designed to combine with the BART generator (see `initial_bart_base_test/`) in a GAN training loop:

```
Source Document
      ↓
BART Generator → candidate summary
                        ↓
          BioBERT Discriminator → faithfulness score
                        ↓
          Adversarial loss (REINFORCE) → backprop to BART
```

The combined loss for BART training is:

```
total_loss = MLE_loss + α * adversarial_loss
```

where `adversarial_loss = 1 - faithfulness_score` and `α` controls the balance between fluency and factual faithfulness.

---

## Requirements

```
Python         3.11
PyTorch        2.0.1+cu117
transformers   4.40.0
datasets       4.x
scikit-learn   1.x
accelerate     1.x
```

Install:
```bash
pip install transformers datasets accelerate scikit-learn
```

---

## Results

| Model | Val Accuracy | Val F1 (Faithful) | Val F1 (Hallucinated) |
|-------|-------------|-------------------|----------------------|
| BioBERT Discriminator (MS2, perturbation negatives) | TBD | TBD | TBD |

Results will be updated after full evaluation runs are complete.

---

## Limitations

- Negative examples are currently simulated via perturbation, not actual model-generated hallucinations. Discriminator performance may not fully reflect real hallucination detection until retrained on BART outputs.
- Labeling all BART-generated summaries as hallucinated (label = 0) is a simplification — BART may generate faithful summaries especially after fine-tuning. A factual verification step (e.g. NLI-based) would improve label quality.
- Input truncation to 512 tokens may lose information from long multi-document sources.

---

## References

- Lee, J. et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics*
- DeYoung, J. et al. (2021). MS²: Multi-Document Summarization of Medical Studies. *EMNLP 2021*
- Lewis, M. et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training. *arXiv:1910.13461*
