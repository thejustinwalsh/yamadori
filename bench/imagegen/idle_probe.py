"""How much CUDA1 an idle sd-server holds after one generation (ttl matters).
Starts sd-server on 127.0.0.1:1240 with smoke_server.FLAGS, one 512x512 image,
then samples CUDA1 used for 20 s of idleness, then kills it."""
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure, smoke_server as S
u0, f0 = measure.smi()
env = dict(os.environ, CUDA_VISIBLE_DEVICES=S.A4000)
p = subprocess.Popen([S.SERVER, *S.FLAGS], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(2)
    idle_before = measure.smi()[0]
    body = json.dumps({"model": "imagegen", "prompt": "a small red cube", "width": 512, "height": 512, "steps": 20, "seed": 1}).encode()
    t = time.time()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{S.PORT}/sdapi/v1/txt2img", data=body, headers={"Content-Type": "application/json"}), timeout=600).read()
    gen = time.time() - t
    samples = []
    for _ in range(10):
        time.sleep(2); samples.append(measure.smi()[0])
finally:
    p.kill(); p.wait()
time.sleep(2)
row = {"probe": "idle_after_generation", "base_used_mib": u0, "loaded_idle_before_mib": idle_before,
       "gen_512_s": round(gen, 1), "idle_after_mib": samples, "after_kill_mib": measure.smi()[0],
       "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(json.dumps(row))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a").write(json.dumps(row) + "\n")
