import { load, sig, eq, dv } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
// u64 -> i64 and f32 -> f32 at the wasm level; the pointer is i32 on wasm32.
sig(sigs, 'checked_mul_u64', ['i64', 'i64', 'i32'], ['i32']);
sig(sigs, 'lerp_f32', ['f32', 'f32', 'f32'], ['f32']);

const M64 = (1n << 64n) - 1n;
const slot = ex.__grader_alloc(8);               // 8-byte aligned
const SENTINEL = 0x0123456789abcdefn;

function mul(a, b) {
  dv(ex).setBigUint64(slot, SENTINEL, true);
  // BigInt arguments are converted to i64 bit patterns; pass the unsigned value.
  const rc = ex.checked_mul_u64(BigInt.asIntN(64, a), BigInt.asIntN(64, b), slot);
  return [rc, dv(ex).getBigUint64(slot, true)];
}
// eq() JSON-serialises arrays, which cannot hold BigInt: compare the parts.
function eqMul([rc, v], [wantRc, wantV], msg) {
  eq(rc, wantRc, `${msg}: status`);
  eq(v, wantV, `${msg}: *out`);
}

// Expected values come from exact BigInt arithmetic, not hand-computed constants.
const pairs = [
  [6n, 7n], [0x1_0000_0001n, 0xffff_fffen], [0xffff_ffffn, 0xffff_ffffn], [M64, 1n],
  [0x1_0000_0001n, 0x1_0000_0000n], [1n << 63n, 2n], [M64, M64], [1n << 32n, 1n << 32n],
  [0n, M64], [0x1234_5678_9abcn, 0xdef0n], [3n, 0x5555_5555_5555_5556n],
];
let fits = 0, overflows = 0;
for (const [a, b] of pairs) {
  const p = a * b;
  const want = p <= M64 ? [0, p] : [1, SENTINEL];
  if (p <= M64) fits++; else overflows++;
  eqMul(mul(a, b), want, `checked_mul_u64(${a}, ${b})`);
}
eq([fits, overflows], [6, 5], 'self-check: the vectors cover both outcomes');
eq(ex.checked_mul_u64(2n, 3n, 0), -1, 'null out');
ex.__grader_free(slot, 8);

const f32 = Math.fround;
function lerpRef(a, b, t) { return f32(f32(a) + f32(f32(f32(b) - f32(a)) * f32(t))); }
for (const [a, b, t] of [[0, 10, 0.25], [1, 2, 0.5], [-3.5, 7.25, 0.1], [1e30, -1e30, 0.3], [0.1, 0.2, 0.7]]) {
  const got = ex.lerp_f32(a, b, t);
  eq(got, lerpRef(a, b, t), `lerp_f32(${a}, ${b}, ${t}) computed in f32`);
}
