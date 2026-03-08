(
echo # Two-Stage BART Pipeline Attempt (PubMed + MS2)
echo.
echo ## Overview
echo Attempted two-stage fine-tuning pipeline: first domain adaptation on PubMed, then fine-tuning on MS2 multi-document summarization.
echo.
echo ## Why This Approach Was Attempted
echo The idea was to first expose BART to medical language via PubMed abstracts, then fine-tune on MS2 for summarization.
echo.
echo ## Known Issues
echo - Colab session resets caused the PubMed checkpoint to be lost before MS2 fine-tuning
echo - MS2 source construction used only study titles instead of full abstracts, causing inverted compression ratios
echo - Final MS2 fine-tune effectively started from base BART, not the PubMed checkpoint
echo.
echo ## Lessons Learned
echo See initial_bart_base_test/ for the corrected pipeline with proper source extraction.
) > "two_stage_pubmed_ms2_attempt\README.md"
