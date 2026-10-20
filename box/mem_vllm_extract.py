import re
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text):
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    lines = text.strip().split("\n")
    start = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b", ln, re.I):
            start = i
            break
    body = "\n".join(lines[start:]).strip()
    if ";" in body:
        body = body[: body.index(";") + 1]
    return body.strip().rstrip(";").strip()


