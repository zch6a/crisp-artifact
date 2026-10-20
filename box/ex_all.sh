#!/bin/bash
# EX evaluation of every generator's outputs on the full task file (run inside CRISP_BOX, default: this directory)
cd "${CRISP_BOX:-$(dirname "$0")}"
for g in qwen2.5-32b-awq gpt56luna dsv4flash qwen2.5-coder_7b llama3.1_8b; do
  echo "=== $g $(date +%T)" >> ex_all.log
  python ex_remote.py tasks_all2.jsonl gen_$g.jsonl flags_all2_$g.json 192 >> ex_all.log 2>&1
done
echo EX_ALL_DONE >> ex_all.log
