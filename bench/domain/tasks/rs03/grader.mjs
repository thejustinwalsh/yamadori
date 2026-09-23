import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, close, eq } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
assert(/export function normalize\(v: Float32Array\): Float32Array;/.test(dts),
  `generated declaration is not normalize(v: Float32Array): Float32Array:\n${dts}`);

const m = createRequire(import.meta.url)(jsPath);
const out = m.normalize(new Float32Array([3, 4]));
assert(out instanceof Float32Array, 'returns a Float32Array');
close(out[0], 0.6, 1e-6, 'x'); close(out[1], 0.8, 1e-6, 'y');
const z = m.normalize(new Float32Array([0, 0, 0]));
eq(Array.from(z), [0, 0, 0], 'zero vector unchanged');
const one = m.normalize(new Float32Array([-2]));
close(one[0], -1, 1e-6, 'single element');
eq(m.normalize(new Float32Array([])).length, 0, 'empty input');
