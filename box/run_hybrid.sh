#!/bin/bash
# hybrid linking (LLM selects columns inside the certified set) with Qwen2.5-32B-Instruct-AWQ (env MODEL overrides)
cd "${CRISP_BOX:-$(dirname "$0")}"
MAXLEN=2112 UTIL=0.94 MAXTOK=256 python mem_text.py hybrid_link_prompts.jsonl hybrid_link_out.jsonl > hybrid_text.log 2>&1
echo HYBRID_LINK_DONE >> hybrid_text.log
