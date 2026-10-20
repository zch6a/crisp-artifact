# Large assets not stored in git

## Distilled block reader (~1.6 GB)

The distilled ModernBERT-large reader used by the fully open pipeline is not stored in git.
Upload it to Hugging Face or Zenodo and fill the URL here:

**URL: <to be filled before submission>**

Expected directory (set `CRISP_MODELS` accordingly):

```
models/laya_block_distill/
  config.json
  model.safetensors
  tokenizer.json / tokenizer_config.json / vocab.txt
```

The released scores in `scores/block_t10_dev_s7.jsonl` already contain this reader's outputs for
BIRD-Dev, so the paper's tables can be reproduced without the weights. The weights are only needed
to score new questions or new schemas with the local reader.

## Datasets

BIRD, Spider, Spider 2.0, OmniSQL and SNAILS are not redistributed here; see the top-level README
for download links. Keep their original licenses.
