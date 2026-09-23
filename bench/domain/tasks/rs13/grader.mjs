import { load, sig, eq, put, GraderFailure } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
// u64 stays a 64-bit wasm value (i64) at the C ABI; pointers and usize are i32 on wasm32.
sig(sigs, 'fnv1a64', ['i32', 'i32'], ['i64']);

// Independent FNV-1a 64 in BigInt, checked against published vectors below.
function fnv(bytes) {
  let h = 0xcbf29ce484222325n;
  for (const b of bytes) {
    h ^= BigInt(b);
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h;
}
const enc = (s) => new TextEncoder().encode(s);
eq(fnv(enc('')), 0xcbf29ce484222325n, 'self-check: FNV-1a("")');
eq(fnv(enc('a')), 0xaf63dc4c8601ec8cn, 'self-check: FNV-1a("a")');
eq(fnv(enc('foobar')), 0x85944171f73967e8n, 'self-check: FNV-1a("foobar")');

function hash(bytes) {
  const p = put(ex, bytes);
  // JS receives an i64 as a signed BigInt; the bits are what matter.
  const r = BigInt.asUintN(64, ex.fnv1a64(p, bytes.length));
  const after = new Uint8Array(ex.memory.buffer, p, bytes.length);
  eq(Array.from(after), Array.from(bytes), 'input bytes untouched');
  ex.__grader_free(p, bytes.length);
  return r;
}

eq(hash(enc('a')), 0xaf63dc4c8601ec8cn, 'fnv1a64("a")');
eq(hash(enc('foobar')), 0x85944171f73967e8n, 'fnv1a64("foobar")');
const big = new Uint8Array(4096);
let s = 12345;
for (let i = 0; i < big.length; i++) { s = (s * 1103515245 + 12345) >>> 0; big[i] = s >>> 24; }
eq(hash(big), fnv(big), 'fnv1a64 over 4 KiB of pseudo-random bytes');
const p = put(ex, enc('xyz'));
eq(BigInt.asUintN(64, ex.fnv1a64(p, 0)), 0xcbf29ce484222325n, 'zero length, non-null pointer');
ex.__grader_free(p, 3);
let r0;
try { r0 = ex.fnv1a64(0, 0); } catch (e) {
  throw new GraderFailure(`fnv1a64(0, 0) trapped (${e.message}): a null pointer with zero length must be accepted`);
}
eq(BigInt.asUintN(64, r0), 0xcbf29ce484222325n, 'null pointer, zero length');
