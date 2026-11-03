# IndicPulse Models Directory

This directory stores fine-tuned model weights.

## MuRIL Fine-tuned Model

**Directory:** `muril_finetuned/`

After fine-tuning on Google Colab (see `notebooks/01_finetune_muril_colab.ipynb`), 
download and place the model here:

```
models/
└── muril_finetuned/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── special_tokens_map.json
```

## Without Fine-tuned Weights

If `muril_finetuned/` doesn't exist, the pipeline automatically falls back to:
1. **Base MuRIL** (`google/muril-base-cased`) with zero-shot classification
2. **Rule-based heuristics** if transformers is not installed

The rule-based fallback is sufficient for demo purposes and produces good results
on clear Hinglish text patterns.

## Training Your Own

See `notebooks/01_finetune_muril_colab.ipynb` for the full Colab fine-tuning notebook.
Requires:
- Google Colab with T4 GPU (free tier)
- Labeled dataset: CSV with `text` and `label` columns (0=Positive, 1=Sarcastic, 2=Neutral, 3=Toxic)
