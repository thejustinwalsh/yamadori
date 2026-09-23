import { load, sig, eq, put, GraderFailure } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
sig(sigs, 'utf8_to_utf16', ['i32', 'i32', 'i32', 'i32'], ['i32']);

const enc = (s) => new TextEncoder().encode(s);
const CANARY = 0xcccc;

/** Convert with a dst of `cap` units followed by 4 canary units. */
function convert(bytes, cap) {
  const src = put(ex, bytes);
  const dst = ex.__grader_alloc((cap + 4) * 2);
  new Uint16Array(ex.memory.buffer, dst, cap + 4).fill(CANARY);
  const r = ex.utf8_to_utf16(src, bytes.length, dst, cap);
  const units = Array.from(new Uint16Array(ex.memory.buffer, dst, cap + 4));
  ex.__grader_free(dst, (cap + 4) * 2);
  ex.__grader_free(src, bytes.length);
  return { r, units };
}
const codeUnits = (s) => Array.from({ length: s.length }, (_, i) => s.charCodeAt(i));

const cases = ['plain ascii', 'héllo', '日本語テキスト',
               'a\u{1F600}b', '\u{1F680}\u{1F30D}\u{10FFFF}', 'mixed é中\u{1F600}!'];
for (const s of cases) {
  const want = codeUnits(s);
  const n = want.length;
  // size query
  const srcBytes = enc(s);
  const qp = put(ex, srcBytes);
  const q = ex.utf8_to_utf16(qp, srcBytes.length, 0, 0);
  ex.__grader_free(qp, srcBytes.length);
  eq(q, n, `required units for ${JSON.stringify(s)} (null dst)`);
  // exact fit
  let { r, units } = convert(enc(s), n);
  eq(r, n, `return value for ${JSON.stringify(s)}`);
  eq(units.slice(0, n), want, `UTF-16 of ${JSON.stringify(s)}`);
  eq(units.slice(n), [CANARY, CANARY, CANARY, CANARY], `nothing written past the ${n} units of ${JSON.stringify(s)}`);
  eq(String.fromCharCode(...units.slice(0, n)), s, 'round-trips through String.fromCharCode');
  // one short: nothing written
  ({ r, units } = convert(enc(s), n - 1));
  eq(r, n, `required units when cap is one short (${JSON.stringify(s)})`);
  if (units.some((u) => u !== CANARY))
    throw new GraderFailure(`cap ${n - 1} < ${n} but dst was written for ${JSON.stringify(s)}: ${units}`);
}

for (const bad of [[0x80], [0x61, 0xe2, 0x82], [0xf8, 0x88, 0x80, 0x80, 0x80]]) {
  const { r, units } = convert(new Uint8Array(bad), 16);
  eq(r, -1, `invalid UTF-8 ${JSON.stringify(bad)}`);
  if (units.some((u) => u !== CANARY)) throw new GraderFailure('invalid UTF-8: dst was written');
}

let r0;
try { r0 = ex.utf8_to_utf16(0, 0, 0, 0); } catch (e) {
  throw new GraderFailure(`utf8_to_utf16(0, 0, 0, 0) trapped (${e.message}): null src with len 0 must be accepted`);
}
eq(r0, 0, 'empty input');
