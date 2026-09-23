import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, eq } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
const m = createRequire(import.meta.url)(jsPath);

assert(/export function rotl64\(\s*\w+: bigint,\s*\w+: number\s*\): bigint;/.test(dts),
  `generated declaration is not rotl64(bigint, number): bigint:\n${dts}`);
assert(/export function popcount64\(\s*\w+: bigint\s*\): number;/.test(dts),
  `generated declaration is not popcount64(bigint): number:\n${dts}`);

const M = (1n << 64n) - 1n;
const rotRef = (x, n) => { const s = BigInt(n % 64); return ((x << s) | (x >> ((64n - s) % 64n))) & M; };

const r = m.rotl64(1n, 1);
assert(typeof r === 'bigint', `rotl64 returns a bigint, got ${typeof r}`);
for (const [x, n] of [[1n, 1], [1n, 63], [1n << 63n, 1], [0x0123456789abcdefn, 4],
                      [0x0123456789abcdefn, 68], [M, 17], [0xf000000000000000n, 0],
                      [0x8000000000000001n, 32], [0xdeadbeefcafebaben, 4000000000]]) {
  const got = m.rotl64(x, n);
  eq(got, rotRef(x, n), `rotl64(0x${x.toString(16)}, ${n})`);
  assert(got >= 0n, `rotl64(0x${x.toString(16)}, ${n}) must be non-negative, got ${got}`);
}

const pc = m.popcount64(M);
assert(typeof pc === 'number', `popcount64 returns a number, got ${typeof pc}`);
eq(pc, 64, 'popcount64(2^64-1)');
eq(m.popcount64(0n), 0, 'popcount64(0)');
eq(m.popcount64(1n << 63n), 1, 'popcount64(2^63)');
eq(m.popcount64(0x8000000000000001n), 2, 'popcount64(2^63 + 1)');
eq(m.popcount64(0x0123456789abcdefn), 32, 'popcount64(0x0123456789abcdef)');
