"""Parse {h, text} vLLM outputs into {h, sql} for box/ex_remote.py.

    python reproduce/parse_text.py --src cache/gen_tag.jsonl --dst cache/gen_tag_sql.jsonl
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "src"))
from jevschema.generator import extract_sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    n = 0
    with open(a.dst, "w", encoding="utf-8") as f:
        for line in open(a.src, encoding="utf-8"):
            r = json.loads(line)
            t = re.sub(r"<think>.*?</think>", "", r.get("text", "") or "", flags=re.S)
            f.write(json.dumps({"h": r["h"], "sql": extract_sql(t)}, ensure_ascii=False) + "\n")
            n += 1
    print(f"parsed {n} -> {a.dst}")


if __name__ == "__main__":
    main()
