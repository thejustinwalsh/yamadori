import { load, sig, eq, assert, view } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
sig(sigs, 'alloc', ['i32'], ['i32']);
sig(sigs, 'dealloc', ['i32', 'i32'], []);
sig(sigs, 'to_upper', ['i32', 'i32', 'i32'], ['i32']);

const live = () => ex.__grader_live_bytes();
const enc = (s) => new TextEncoder().encode(s);
const dec = (b) => new TextDecoder('utf-8', { fatal: true }).decode(b);

/** Full protocol: alloc input, write, to_upper, read, dealloc both. */
function upper(bytes) {
  const p = ex.alloc(bytes.length);
  new Uint8Array(ex.memory.buffer, p, bytes.length).set(bytes);
  const lenp = ex.__grader_alloc(4);           // caller-owned usize slot
  new DataView(ex.memory.buffer).setUint32(lenp, 0xdeadbeef, true);
  const r = ex.to_upper(p, bytes.length, lenp) >>> 0;
  const n = new DataView(ex.memory.buffer).getUint32(lenp, true);
  eq(Array.from(view(ex, p, bytes.length)), Array.from(bytes), 'to_upper left the input untouched');
  let out = null;
  if (r !== 0) {
    out = view(ex, r, n);
    ex.dealloc(r, n);
  } else {
    eq(n, 0, '*out_len when returning null');
  }
  ex.__grader_free(lenp, 4);
  ex.dealloc(p, bytes.length);
  return out;
}

const base = live();

// alloc/dealloc on their own balance, and the memory is writable.
for (const n of [0, 1, 7, 64, 1000]) {
  const p = ex.alloc(n);
  new Uint8Array(ex.memory.buffer, p, n).fill(0xab);
  ex.dealloc(p, n);
}
eq(live() - base, 0, 'live bytes after alloc/dealloc pairs');

const cases = [
  'hello, world',
  '',
  'straße',                  // sharp s -> SS
  'ŉ and ᾀ',           // n-apostrophe and a Greek letter that expand
  'été à škoda физика',
  'ﬃ ligature',             // ffi ligature -> FFI
  '\u{1F600} unchanged emoji',
  'ΐ'.repeat(50),           // each becomes three code points
];
for (const s of cases) {
  const before = live();
  const out = upper(enc(s));
  assert(out !== null, `to_upper(${JSON.stringify(s)}) returned null for valid UTF-8`);
  eq(dec(out), s.toUpperCase(), `to_upper(${JSON.stringify(s)})`);
  eq(live() - before, 0, `bytes leaked by to_upper(${JSON.stringify(s)}) after dealloc(ptr, *out_len)`);
}

for (const bad of [[0xff], [0x61, 0xc3], [0xed, 0xa0, 0x80]]) {
  const before = live();
  eq(upper(new Uint8Array(bad)), null, `to_upper of invalid UTF-8 ${JSON.stringify(bad)}`);
  eq(live() - before, 0, 'bytes leaked on the invalid-UTF-8 path');
}

eq(live() - base, 0, 'live bytes at the end');
