import { load, sig, eq, close, put, dv } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
sig(sigs, 'stats_size', [], ['i32']);
sig(sigs, 'stats_compute', ['i32', 'i32', 'i32'], ['i32']);

// C layout on wasm32: count @0 (u32), 4 bytes padding, min @8, max @16, mean @24; size 32.
eq(ex.stats_size(), 32, 'sizeof(struct Stats) on wasm32');

const out = ex.__grader_alloc(32);
function fillSentinel() {
  const d = dv(ex);
  for (let i = 0; i < 32; i += 4) d.setUint32(out + i, 0xa5a5a5a5, true);
}
function read() {
  const d = dv(ex);
  return { count: d.getUint32(out, true), min: d.getFloat64(out + 8, true),
           max: d.getFloat64(out + 16, true), mean: d.getFloat64(out + 24, true) };
}
function compute(xs) {
  const p = put(ex, new Uint8Array(new Float64Array(xs).buffer));
  fillSentinel();
  const rc = ex.stats_compute(p, xs.length, out);
  ex.__grader_free(p, xs.length * 8);
  return rc;
}

for (const xs of [[3, 1, 4, 1, 5, 9, 2, 6], [2.5], [10, 20, 30], [-1.5, -7.25, -0.5],
                  Array.from({ length: 1000 }, (_, i) => Math.sin(i) * 100 + 50)]) {
  eq(compute(xs), 0, `status for ${xs.length} values`);
  const s = read();
  eq(s.count, xs.length, 'count (u32 at offset 0)');
  eq(s.min, Math.min(...xs), 'min (f64 at offset 8)');
  eq(s.max, Math.max(...xs), 'max (f64 at offset 16)');
  close(s.mean, xs.reduce((a, b) => a + b, 0) / xs.length, 1e-9, 'mean (f64 at offset 24)');
}

const untouched = () => Array.from(new Uint32Array(ex.memory.buffer, out, 8)).every((w) => w === 0xa5a5a5a5);
eq(compute([]), 1, 'len 0 returns 1');
eq(untouched(), true, 'len 0 leaves *out untouched');
fillSentinel();
eq(ex.stats_compute(0, 0, out), 1, 'null values with len 0 returns 1');
eq(untouched(), true, 'null values, len 0: *out untouched');
eq(ex.stats_compute(0, 4, out), -1, 'null values with len > 0');
eq(untouched(), true, 'null values: *out untouched');
const p = put(ex, new Uint8Array(new Float64Array([1, 2]).buffer));
eq(ex.stats_compute(p, 2, 0), -1, 'null out');
ex.__grader_free(p, 16);
ex.__grader_free(out, 32);
