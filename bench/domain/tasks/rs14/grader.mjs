import { load, sig, eq, put, GraderFailure } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
sig(sigs, 'count_chars', ['i32', 'i32'], ['i32']);

function count(bytes) {
  const p = put(ex, bytes);
  const r = ex.count_chars(p, bytes.length);
  eq(Array.from(new Uint8Array(ex.memory.buffer, p, bytes.length)), Array.from(bytes),
     'input bytes untouched');
  ex.__grader_free(p, bytes.length);
  return r;
}
const enc = (s) => new TextEncoder().encode(s);

for (const s of ['hello', 'héllo wörld', '日本語', 'a\u{1F600}b\u{1F680}',
                 '\u0000nul inside', 'é (combining accent = 2 chars)']) {
  eq(count(enc(s)), [...s].length, `count_chars(${JSON.stringify(s)})`);
}
const big = 'xé中\u{1F600}'.repeat(500);
eq(count(enc(big)), 2000, 'count_chars over 5 KiB of mixed-width text');

for (const [name, bad] of [['lone continuation byte', [0x61, 0x80, 0x62]],
                           ['truncated 3-byte sequence', [0xe4, 0xb8]],
                           ['overlong encoding of "/"', [0xc0, 0xaf]],
                           ['UTF-16 surrogate encoded in UTF-8', [0xed, 0xa0, 0x80]],
                           ['0xFF byte', [0x41, 0xff]]]) {
  eq(count(new Uint8Array(bad)), -1, `invalid UTF-8 (${name})`);
}
let r0;
try { r0 = ex.count_chars(0, 0); } catch (e) {
  throw new GraderFailure(`count_chars(0, 0) trapped (${e.message}): a null pointer with zero length must be accepted`);
}
eq(r0, 0, 'null pointer, zero length');
const p = put(ex, enc('abc'));
eq(ex.count_chars(p, 0), 0, 'zero length');
eq(ex.count_chars(p, 2), 2, 'prefix of the buffer');
ex.__grader_free(p, 3);
