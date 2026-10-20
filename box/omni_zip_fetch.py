"""Fetch selected members of OmniSQL-datasets data.zip (22 GB) with parallel HTTP range requests.
  python omni_zip_fetch.py list <regex>            -> prints matching members (central directory cached in zip_cd.bin)
  python omni_zip_fetch.py get <regex> <outdir>    -> extracts matching members
Only the 60 MB central directory plus the selected members are downloaded."""
import os, re, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
import requests

U0 = os.environ.get("HF_ENDPOINT", "https://huggingface.co") + "/datasets/seeklhy/OmniSQL-datasets/resolve/main/data.zip"
SIZE = 22214530701
CHUNK, NPAR = 4 << 20, int(os.environ.get("NPAR", "32"))
S = requests.Session()


def url():
    r = S.head(U0, allow_redirects=False, timeout=30)
    return r.headers.get("location", U0)


U = url()


def rng(a, b):  # inclusive a..b
    for _ in range(8):
        try:
            r = S.get(U, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
        except Exception:
            pass
    raise RuntimeError(f"range {a}-{b} failed")


def fetch(a, n):
    parts = [(s, min(s + CHUNK, a + n) - 1) for s in range(a, a + n, CHUNK)]
    with ThreadPoolExecutor(NPAR) as ex:
        return b"".join(ex.map(lambda p: rng(*p), parts))


def central():
    if os.path.exists("zip_cd.bin"):
        return open("zip_cd.bin", "rb").read()
    tail = rng(SIZE - 65536, SIZE - 1)
    i = tail.rfind(b"PK\x06\x06")
    n, sz, off = struct.unpack("<QQQ", tail[i + 32:i + 56])
    print(f"entries {n} cd {sz/1e6:.1f} MB", flush=True)
    cd = fetch(off, sz)
    open("zip_cd.bin", "wb").write(cd)
    return cd


def entries(cd):
    p = 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p + 10:p + 12])
        csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        ex = cd[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q + 4 <= len(ex):  # zip64 extra
            hid, hl = struct.unpack("<HH", ex[q:q + 4])
            if hid == 1:
                vals, k = [], q + 4
                for flag in (usz == 0xFFFFFFFF, csz == 0xFFFFFFFF, off == 0xFFFFFFFF):
                    if flag:
                        vals.append(struct.unpack("<Q", ex[k:k + 8])[0]); k += 8
                it = iter(vals)
                if usz == 0xFFFFFFFF: usz = next(it)
                if csz == 0xFFFFFFFF: csz = next(it)
                if off == 0xFFFFFFFF: off = next(it)
            q += 4 + hl
        yield name, meth, csz, usz, off
        p += 46 + nl + el + cl


def extract(e, outdir):
    name, meth, csz, usz, off = e
    if name.endswith("/"):
        return
    h = rng(off, off + 29)
    nl, el = struct.unpack("<HH", h[26:30])
    data = fetch(off + 30 + nl + el, csz) if csz else b""
    if meth == 8:
        data = zlib.decompress(data, -15)
    path = os.path.join(outdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(data)


if __name__ == "__main__":
    mode, pat = sys.argv[1], re.compile(sys.argv[2])
    E = [e for e in entries(central()) if pat.search(e[0])]
    print(f"{len(E)} match, {sum(e[2] for e in E)/1e6:.1f} MB compressed", flush=True)
    if mode == "list":
        for e in E[:300]:
            print(e[0], e[3])
    else:
        for k, e in enumerate(E):
            extract(e, sys.argv[3])
            if k % 50 == 0 or e[2] > 50e6:
                print("got", k + 1, e[0], flush=True)
        print("done", flush=True)
