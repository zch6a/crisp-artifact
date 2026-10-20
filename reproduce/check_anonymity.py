"""Fail if the repository contains identity-revealing absolute paths or emails.

    python reproduce/check_anonymity.py

Scans source, docs and configs (not the numeric score files) for drive-letter paths,
POSIX home/scratch paths, emails and private keys. Exit code 1 on any hit.
"""
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
POSIX = re.compile(r"/(?:home|mnt|root|Users|volume|autodl-tmp|workspace)/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEY = re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY")
SKIP_DIRS = {"scores", "results", "__pycache__", ".git"}
EXTS = {".py", ".md", ".txt", ".sh", ".json", ".example", ".cfg", ".yml", ".yaml"}


def main():
    hits = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() not in EXTS and f != "LICENSE":
                continue
            p = os.path.join(d, f)
            try:
                text = open(p, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if "http" in line:
                    continue
                if DRIVE.search(line) or POSIX.search(line) or EMAIL.search(line) or KEY.search(line):
                    hits.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:120]}")
    if hits:
        print("potential identity leaks:")
        print("\n".join(hits))
        return 1
    print("no absolute paths, emails or private keys found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
