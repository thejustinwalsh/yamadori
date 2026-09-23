import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, eq } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
const m = createRequire(import.meta.url)(jsPath);

assert(typeof m.xorCipher === 'function',
  `no export named xorCipher (exports: ${Object.keys(m).join(', ')})`);
// Parameter names are the answer's choice; the types are the contract.
assert(/export function xorCipher\(\s*\w+: Uint8Array,\s*\w+: Uint8Array\s*\): Uint8Array;/.test(dts),
  `generated declaration is not xorCipher(Uint8Array, Uint8Array): Uint8Array:\n${dts}`);

const data = new Uint8Array([0x00, 0xff, 0x10, 0x20, 0x30, 0x41, 0x42]);
const key = new Uint8Array([0x0f, 0xf0]);
const out = m.xorCipher(data, key);
assert(out instanceof Uint8Array, 'returns a Uint8Array');
eq(Array.from(out), [0x0f, 0x0f, 0x1f, 0xd0, 0x3f, 0xb1, 0x4d], 'key repeats across the data');
eq(Array.from(data), [0x00, 0xff, 0x10, 0x20, 0x30, 0x41, 0x42], 'data not modified');
eq(Array.from(key), [0x0f, 0xf0], 'key not modified');
assert(out.buffer !== data.buffer, 'a new array is returned');

eq(Array.from(m.xorCipher(out, key)), Array.from(data), 'applying twice round-trips');
eq(Array.from(m.xorCipher(data, new Uint8Array([]))), Array.from(data), 'empty key -> copy');
eq(m.xorCipher(new Uint8Array([]), key).length, 0, 'empty data');
const long = new Uint8Array(10000).map((_, i) => i & 0xff);
const lk = new Uint8Array([1, 2, 3]);
const lo = m.xorCipher(long, lk);
eq(lo.length, 10000, 'length preserved');
eq(lo[9999], (9999 & 0xff) ^ lk[9999 % 3], 'last byte of a long input');
