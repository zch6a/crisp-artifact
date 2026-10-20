"""Apply our kept column sets to OmniSQL's OFFICIAL BIRD-Dev inputs (data/dev_bird.json input_seq: DDL with
question-matched example values and column descriptions).  The official text is kept verbatim for every kept
column; non-kept column lines are removed, tables with no kept column are dropped, PRIMARY KEY / FOREIGN KEY lines
are kept only when all their local columns are kept (FK: and the referenced table is present).
  python official_prune.py <name> <prompts.jsonl,...> <tasks.jsonl,...> [sys1,sys2,...]
-> queue2/<name>.jsonl, tasks_<name>.jsonl (systems prefixed 'off:'); 'off:full' is the unmodified official input.
A column counts as kept when our prompt's block for that table has a line starting with '  <column> '."""
import hashlib, json, re, sys

OFF = None
name = pfiles = tfiles = None
only = None


def ident(s):
    s = s.strip()
    if s.startswith("`"):
        return s[1:s.index("`", 1)]
    if s.startswith('"'):
        return s[1:s.index('"', 1)]
    return re.split(r"[\s(]", s, 1)[0]


def our_blocks(prompt):
    """{table_lower: [column-line strings]} from our DDL-style schema text."""
    B, cur = {}, None
    for line in prompt.split("\n"):
        m = re.match(r"CREATE TABLE (.+?) \($", line)
        if m:
            cur = m.group(1).strip().lower(); B[cur] = []; continue
        if cur is not None:
            if line.startswith(");"):
                cur = None
            else:
                B[cur].append(line.lower())
    return B


def kept_cols(blocks, table, col):
    lines = blocks.get(table.lower())
    if lines is None:
        return False
    c = col.lower()
    return any(l.startswith(f"  {c} ") or l.rstrip(",") == f"  {c}" for l in lines)


def prune(off, blocks):
    a = off.index("Database Schema:\n") + len("Database Schema:\n")
    b = off.index("\nThis schema describes")
    schema = off[a:b]
    out_tables, present, stats = [], set(), [0, 0]
    raw = [t for t in re.split(r"\n\n(?=CREATE TABLE)", schema.strip()) if t.strip()]
    parsed = []
    for blk in raw:
        lines = blk.split("\n")
        tname = ident(lines[0][len("CREATE TABLE "):].rstrip(" ("))
        items = [l for l in lines[1:] if l.strip() not in (");", ")")]
        parsed.append((tname, lines[0], items))
    for tname, head, items in parsed:
        if tname.lower() in blocks:
            present.add(tname.lower())
    for tname, head, items in parsed:
        if tname.lower() not in present:
            continue
        keep, cols_here = [], set()
        for it in items:
            s = it.strip()
            if s.upper().startswith(("PRIMARY KEY", "CONSTRAINT", "FOREIGN KEY", "UNIQUE")):
                continue
            stats[1] += 1
            c = ident(s)
            if kept_cols(blocks, tname, c):
                keep.append(it); cols_here.add(c.lower()); stats[0] += 1
        if not keep:
            continue
        for it in items:
            s = it.strip(); u = s.upper()
            if u.startswith("PRIMARY KEY"):
                pk = [ident(x) .lower() for x in s[s.index("(") + 1:s.rindex(")")].split(",")]
                if all(x in cols_here for x in pk):
                    keep.append(it)
            elif "FOREIGN KEY" in u:
                m = re.search(r"FOREIGN KEY \((.+?)\) REFERENCES (.+?) \(", s)
                if m:
                    loc = [ident(x).lower() for x in m.group(1).split(",")]
                    if all(x in cols_here for x in loc) and ident(m.group(2)).lower() in present:
                        keep.append(it)
        # re-punctuate: separator comma sits before the ' -- ' comment (or at line end)
        body = []
        for k, it in enumerate(keep):
            d, sep, cm = it.partition(" -- ")
            d = d.rstrip().rstrip(",")
            d = d + ("," if k < len(keep) - 1 else "")
            body.append(d + (" -- " + cm if sep else ""))
        out_tables.append(head + "\n" + "\n".join(body) + "\n);")
    return off[:a] + "\n\n".join(out_tables) + "\n" + off[b:], stats


