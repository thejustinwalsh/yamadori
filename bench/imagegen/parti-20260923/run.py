"""PartiPrompts sample through the live Images API (:1234 -> llama-swap imagegen).

One prompt per category from the 8 largest PartiPrompts categories, drawn with
seed 20260923: nobody picks the prompts, and every result is kept.
"""
import csv
import json
import os
import random
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "parti")
os.makedirs(OUT, exist_ok=True)
key = open(os.path.join(HERE, "dogfood.key")).read().strip()

rows = list(csv.DictReader(open(os.path.join(HERE, "PartiPrompts.tsv"), encoding="utf-8"), delimiter="\t"))
cats = ["Animals", "Artifacts", "World Knowledge", "People", "Outdoor Scenes",
        "Illustrations", "Vehicles", "Food & Beverage"]
rng = random.Random(20260923)
picks = [rng.choice([r for r in rows if r["Category"] == c]) for c in cats]
manifest = []
for i, r in enumerate(picks):
    body = {"model": "yamadori", "prompt": r["Prompt"], "size": "1024x1024",
            "response_format": "b64_json"}
    req = urllib.request.Request("http://127.0.0.1:1234/v1/images/generations",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        import base64
        png = base64.b64decode(d["data"][0]["b64_json"])
        path = os.path.join(OUT, f"{i:02d}.png")
        open(path, "wb").write(png)
        ok, err = True, None
    except Exception as e:  # noqa: BLE001
        ok, err, path = False, f"{type(e).__name__}: {e}"[:300], None
    row = {"i": i, "category": r["Category"], "challenge": r["Challenge"],
           "prompt": r["Prompt"], "ok": ok, "error": err, "file": path,
           "seconds": round(time.time() - t0, 1)}
    manifest.append(row)
    print(json.dumps(row), flush=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1)
print("done", sum(m["ok"] for m in manifest), "of", len(manifest))
