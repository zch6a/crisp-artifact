"""List / extract members of OmniSQL-datasets data.zip over HTTP range requests (no full 22 GB download).
  python omni_zip_list.py list <regex>      python omni_zip_list.py get <regex> <outdir>"""
import os, re, sys
from remotezip import RemoteZip
U = os.environ.get("HF_ENDPOINT", "https://huggingface.co") + "/datasets/seeklhy/OmniSQL-datasets/resolve/main/data.zip"
mode, pat = sys.argv[1], re.compile(sys.argv[2])
with RemoteZip(U) as z:
    L = [i for i in z.infolist() if pat.search(i.filename)]
    print(f"{len(L)} members match, {sum(i.file_size for i in L)/1e6:.1f} MB", flush=True)
    if mode == "list":
        for i in L[:200]:
            print(i.filename, i.file_size)
    else:
        for n, i in enumerate(L):
            z.extract(i, sys.argv[3])
            if n % 20 == 0:
                print("extracted", n + 1, i.filename, flush=True)
        print("done", flush=True)
