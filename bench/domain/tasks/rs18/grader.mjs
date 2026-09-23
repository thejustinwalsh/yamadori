import { load, sig, eq, assert, put, view, dv } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
sig(sigs, 'interner_new', [], ['i32']);
sig(sigs, 'interner_intern', ['i32', 'i32', 'i32'], ['i32']);
sig(sigs, 'interner_get', ['i32', 'i32', 'i32'], ['i32']);
sig(sigs, 'interner_len', ['i32'], ['i32']);
sig(sigs, 'interner_free', ['i32'], []);

const U32_MAX = 0xffffffff;
const enc = (s) => new TextEncoder().encode(s);
const dec = (b) => new TextDecoder().decode(b);
const live = () => ex.__grader_live_bytes();

// The caller's buffer is scribbled over and freed right after each call.
function intern(h, bytes) {
  const p = put(ex, bytes);
  const id = ex.interner_intern(h, p, bytes.length) >>> 0;
  new Uint8Array(ex.memory.buffer, p, bytes.length).fill(0x7e); // '~'
  ex.__grader_free(p, bytes.length);
  return id;
}
const lenSlot = ex.__grader_alloc(4);
function get(h, id) {
  dv(ex).setUint32(lenSlot, 0xdeadbeef, true);
  const p = ex.interner_get(h, id, lenSlot) >>> 0;
  const n = dv(ex).getUint32(lenSlot, true);
  return p === 0 ? { p, n, s: null } : { p, n, s: dec(view(ex, p, n)) };
}

const base = live();
const h = ex.interner_new();
assert(h !== 0, 'interner_new returned null');
eq(ex.interner_len(h), 0, 'new interner is empty');

eq(intern(h, enc('alpha')), 0, 'first id');
eq(intern(h, enc('beta')), 1, 'second id');
eq(intern(h, enc('alpha')), 0, 'equal string -> existing id');
eq(intern(h, enc('')), 2, 'empty string is a string');
eq(intern(h, enc('été \u{1F600}')), 3, 'non-ASCII string');
eq(ex.interner_len(h), 4, 'distinct strings');

const first = get(h, 0);
eq(first.s, 'alpha', 'get(0) after the caller overwrote and freed its buffer');
eq(first.n, 5, 'length of id 0');
eq(get(h, 1).s, 'beta', 'get(1)');
eq(get(h, 2).n, 0, 'empty string length');
eq(get(h, 3).s, 'été \u{1F600}', 'get(3)');

// Grow a lot; earlier pointers must not move and must still read correctly.
for (let i = 0; i < 2000; i++) intern(h, enc(`word-${i}-${'x'.repeat(i % 37)}`));
eq(ex.interner_len(h), 2004, 'after 2000 more strings');
eq(intern(h, enc('word-1234-' + 'x'.repeat(1234 % 37))), 4 + 1234, 'dedupe after growth');
const again = get(h, 0);
eq(again.p, first.p, 'the pointer for id 0 stayed the same while interning more strings');
eq(dec(view(ex, first.p, 5)), 'alpha', 'bytes at the original pointer are still "alpha"');
eq(get(h, 1999 + 4).s, `word-1999-${'x'.repeat(1999 % 37)}`, 'last id');

// Rejections.
eq(get(h, 2004).p, 0, 'unknown id -> null');
eq(dv(ex).getUint32(lenSlot, true), 0xdeadbeef, 'unknown id: out_len untouched');
eq(ex.interner_get(h, 0, 0), 0, 'null out_len -> null');
eq(ex.interner_get(0, 0, lenSlot), 0, 'null handle -> null');
eq(intern(h, new Uint8Array([0x61, 0xff])), U32_MAX, 'invalid UTF-8 -> u32::MAX');
eq(ex.interner_intern(h, 0, 3) >>> 0, U32_MAX, 'null ptr with len > 0 -> u32::MAX');
eq(ex.interner_intern(h, 0, 0) >>> 0, 2, 'null ptr with len 0 is the empty string');
eq(ex.interner_intern(0, 0, 0) >>> 0, U32_MAX, 'null handle -> u32::MAX');
eq(ex.interner_len(0), 0, 'len of null');
eq(ex.interner_len(h), 2004, 'rejections added nothing');

ex.interner_free(h);
ex.interner_free(0);
eq(live() - base, 0, 'bytes still live after interner_free (leak)');
ex.__grader_free(lenSlot, 4);
