# SumMedi — Medical Text Summarization

Recent transformer models such as BART and PEGASUS have improved abstractive text summarization, but they often face factual hallucinations, especially in high-stakes domains like biomedical text. Previous adversarial approaches for summarization have shown improvements in general summary quality, but most rely on pre-transformer architectures like LSTM/CNN and focus on news datasets rather than domain specific factual consistency.

To address this limitation, this work proposes a GAN-transformer framework for single document medical summarization. A pretrained transformer (BART or PEGASUS) serves as the generator, while a BioBERT-based domain specific discriminator evaluates factual consistency between the source document and the generated summary. The generator is trained using both maximum likelihood and adversarial objectives, encouraging summaries that are fluent and factually aligned. This approach aims to reduce hallucination and improve reliability in biomedical summarization through quantitative evaluation metrics like ROUGE scores and qualitative analysis.

This is a research project exploring transformer-based approaches to automatic summarization of medical literature, with a focus on multi-document clinical evidence synthesis.

---

## Overview

Medical literature is vast and growing rapidly. Clinicians and researchers struggle to synthesize evidence across multiple studies efficiently. SumMedi investigates whether fine-tuned transformer models wrapped in GAN architectures can produce accurate, coherent summaries of medical texts and reduce hallucination during text generation.

This repository aims to contain the entire Generator Adversarial Network architecture (BART as Generator and BioBERT as Discriminator) to summarize medical text documents. BART known for its text generation capabilities will generate best abstractive summaries while BioBERT which is domain specific in bio-medical data will help judge and evaluate the summaries.
Together, BART (generator) and BioBERT (discriminator) wrapped in a GAN architecture will complement each other's abilities to generate accurate summaries with drastic reduction in hallucinations.

---

## Repository Structure

```
SumMedi/
├── initial_bart_base_test/          # BART fine-tuned on MS2 (abstractive summarization)
│   ├── bart_ms2_training.ipynb      # Full training pipeline
│   ├── final_model/                 # Model config & tokenizer (weights shared separately)
│   ├── ms2_train_processed.csv      # Cleaned training pairs
│   ├── ms2_val_processed.csv        # Cleaned validation pairs
│   ├── ms2_test_processed.csv       # Cleaned test pairs
│   └── README.md                    # Detailed notes for this approach
│
├── two_stage_pubmed_ms2_attempt/    # Attempted 2-stage pipeline (PubMed → MS2)
│   ├── two_stage_bart_ms2.ipynb     # Training notebook with documented issues
│   └── README.md                    # What was attempted and why it didn't work
│
└── README.md                        # This file
```

---

## Models

### BART
- **Generator model:** `facebook/bart-base`
- **Dataset:** MS2 multi-document summarization dataset
- **Task:** Abstractive summarization of systematic review evidence
- **Approach:** Fine-tune BART on properly extracted source/target pairs from MS2, where source = concatenated study abstracts from included clinical trials, and target = the Cochrane review abstract

### BioBERT
- Domain-adapted BERT model pre-trained on biomedical text
- Used as discriminator in GAN architecture to classify real and generated summaries.


---

## Dataset

**MS2 — Multi-Document Summarization of Medical Studies**

A large-scale dataset of Cochrane systematic reviews, where each review summarizes evidence from multiple included studies.

- ~180,000 training reviews
- Source: included study titles and abstracts
- Target: the Cochrane review abstract
- Download: [Kaggle — mathurinache/ms2multidocuments](https://www.kaggle.com/datasets/mathurinache/ms2multidocuments)

The raw dataset files are not included in this repository due to size. Download and place the three `.jsonl` files in a local `ms2_data/` folder before running any notebooks.

---

## Key Design Decisions

**Why MS2?**
MS2 is one of the few datasets designed specifically for multi-document medical summarization. Each target summary (Cochrane abstract) is written by domain experts synthesizing evidence from multiple RCTs and observational studies — making it high quality and clinically grounded.

**Why BART?**
BART (Lewis et al., 2019) is a denoising autoencoder pre-trained for sequence-to-sequence tasks. Its architecture makes it particularly well-suited for abstractive summarization compared to encoder-only models like BERT.

**Source construction**
A critical design choice was how to construct the multi-document source input. Early experiments (see `two_stage_pubmed_ms2_attempt/`) used only study titles, which produced inverted compression ratios and degraded model quality. The corrected pipeline concatenates full study abstracts with titles, producing meaningful compression ratios of 0.05–0.30.

---

## Requirements

```
Python         3.11
PyTorch        2.0.1+cu117
transformers   4.40.0
datasets       4.x
evaluate       0.4.x
accelerate     1.x
sentencepiece
rouge_score
```

Install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.40.0 datasets evaluate accelerate sentencepiece rouge_score
```

---

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| BART-base (MS2) | TBD | TBD | TBD |
| BioBERT | TBD | TBD | TBD |

Results will be updated after full evaluation runs are complete.

---

## References

- Lewis, M. et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *arXiv:1910.13461*
- DeYoung, J. et al. (2021). MS²: Multi-Document Summarization of Medical Studies. *EMNLP 2021*
- Lee, J. et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics*
