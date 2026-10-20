import json, os, sys, sqlite3, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
from exec_eval import execute_sql
DB = BOX + "/bird_dev_databases"  # flat copy: <db_id>.sqlite for every BIRD-Dev database
seen = {}
for l in open(BOX + "/tasks_all2.jsonl"):
    t = json.loads(l)
    if t["sys"] != "full": continue
    t0 = time.time(); rows, err = execute_sql(f"{DB}/{t['db']}.sqlite", t["gold"], 30.0)
    dt = time.time() - t0
    if err or dt > 10: print(t["i"], t["db"], f"{dt:.1f}s", err[:80])
print("sqlite", sqlite3.sqlite_version)
