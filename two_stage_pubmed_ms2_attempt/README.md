# Two-Stage BART Pipeline Attempt (PubMed + MS2)

## Overview
Attempted two-stage fine-tuning pipeline: first domain adaptation on PubMed, then fine-tuning on MS2 multi-document summarization.

## Why This Approach Was Attempted
The idea was to first expose BART to medical language via PubMed abstracts, then fine-tune on MS2 for summarization.

## Known Issues
- Colab session resets caused the PubMed checkpoint to be lost before MS2 fine-tuning
- MS2 source construction used only study titles instead of full abstracts, causing inverted compression ratios
- Final MS2 fine-tune effectively started from base BART, not the PubMed checkpoint

## Lessons Learned
See initial_bart_base_test/ for the corrected pipeline with proper source extraction.

