#!/bin/bash
# generate all prompts with OmniSQL-7B (vLLM, greedy)
cd "${CRISP_BOX:-$(dirname "$0")}"
M="${CRISP_MODELS:-../models}/OmniSQL-7B"
TP=1 UTIL=0.93 MAXLEN=8192 python mem_vllm_plain.py "$M" prompts_omni.jsonl gen_omnisql7b.jsonl >> omni_gen.log 2>&1
echo OMNI_GEN_DONE >> omni_gen.log
