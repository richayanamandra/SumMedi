# BART-base Fine-tuned on MS2

Fine-tuning `facebook/bart-base` on the MS2 multi-document medical summarization dataset. This is the primary BART pipeline for the SumMedi project.

---

## What This Does

Takes multiple clinical study abstracts as input and generates a single coherent summary in the style of a Cochrane systematic review abstract.

**Input (source):** Concatenated titles and abstracts of included studies from a systematic review

**Output (target):** A Cochrane-style abstract summarizing the evidence

---

## Files

| File | Description |
|------|-------------|
| `bart_ms2_training.ipynb` | Full pipeline — data loading, preprocessing, tokenization, training, evaluation, inference |
| `ms2_train_processed.csv` | ~4,000–7,000 cleaned source/target training pairs after filtering |
| `ms2_val_processed.csv` | Validation split |
| `ms2_test_processed.csv` | Test split |
| `final_model/` | Model config and tokenizer files. Model weights (`model.safetensors`) shared separately — see below |

---

## How to Run

**1. Download the MS2 dataset from Kaggle:**
```
https://www.kaggle.com/datasets/mathurinache/ms2multidocuments
```
Place the three `.jsonl` files in a local folder and update `MS2_DATA_DIR` in Cell 2 of the notebook.

**2. Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.40.0 datasets evaluate accelerate sentencepiece rouge_score numpy"<2"
```

**3. Run cells in order:**
- Cell 1 — Verify GPU
- Cell 2 — Set paths
- Cell 3 — Load and flatten MS2 data
- Cell 4 — Load BART and tokenize
- Cell 5 — Train
- Cell 6 — Save model
- Cell 7 — Evaluate on test set
- Cell 8 — Run inference on custom input

**4. To load the trained model weights:**

Download `model.safetensors` from the shared Drive link and place it in the `final_model/` folder alongside the existing config files. Then load with:

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model     = BartForConditionalGeneration.from_pretrained("./final_model")
tokenizer = BartTokenizer.from_pretrained("./final_model")
```

---

## Data Pipeline

The MS2 dataset required careful source construction. Each record in MS2 is a Cochrane systematic review with a list of included studies. The naive approach of using only study titles (as done in the earlier two-stage attempt) produces sources shorter than their targets — meaning the model is asked to expand text rather than compress it, which is the wrong task entirely.

The corrected approach:

```
For each included study reference:
    source += study_title + ". " + study_abstract

target = cochrane_review_abstract
```

After filtering records where `len(source_words) > len(target_words)`, the dataset has a mean compression ratio of ~0.10–0.25, which is appropriate for summarization.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | facebook/bart-base |
| Max input length | 512 tokens |
| Max target length | 192 tokens |
| Learning rate | 3e-5 |
| Effective batch size | 16 (batch 2 × grad accum 8) |
| Epochs | 6 (early stopping patience 3) |
| Scheduler | Cosine |
| Precision | fp16 |
| Hardware | NVIDIA RTX 2050 (4GB VRAM) |

---

## Known Limitations

- `bart-base` (139M parameters) is the lightest BART variant. Using `bart-large-cnn` as a starting point would likely yield better results as it is pre-trained specifically for summarization.
- 512 token input truncates longer multi-document sources. Many MS2 sources exceed this limit. A longer-context model or hierarchical encoding would be needed to capture the full evidence.
- 4GB VRAM constrains batch size and input length significantly.

---

## What Was Fixed vs. the Earlier Attempt

The `two_stage_pubmed_ms2_attempt/` folder contains an earlier Colab-based pipeline that had two fundamental problems. First, the source was constructed from study titles only, which meant average source length (~344 words) was shorter than average target length (~385 words) — an inverted compression ratio that trained the model to hallucinate rather than summarize. Second, the two-stage design (PubMed pre-training followed by MS2 fine-tuning) failed because the Colab session reset between stages, causing the PubMed checkpoint to be lost and the MS2 fine-tuning to silently restart from base BART weights.

This pipeline fixes both issues: proper abstract extraction for the source, and a single clean fine-tuning run on MS2 from `facebook/bart-base`.
