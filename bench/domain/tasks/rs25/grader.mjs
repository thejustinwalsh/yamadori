import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, eq, close } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
const m = createRequire(import.meta.url)(jsPath);

assert(typeof m.summarize === 'function', `no export named summarize (exports: ${Object.keys(m).join(', ')})`);

function check(xs) {
  const r = m.summarize(new Float64Array(xs));
  assert(r !== null && typeof r === 'object', `summarize(${xs.length} values) returned ${r}`);
  assert(Object.getPrototypeOf(r) === Object.prototype,
    `result is not a plain object (prototype is ${Object.getPrototypeOf(r)?.constructor?.name})`);
  eq(Object.keys(r).sort(), ['count', 'max', 'mean', 'min'], 'own enumerable keys');
  for (const k of ['count', 'min', 'max', 'mean']) {
    const d = Object.getOwnPropertyDescriptor(r, k);
    assert(d && 'value' in d && typeof d.value === 'number', `${k} is an own data property holding a number`);
  }
  eq(r.count, xs.length, 'count');
  eq(r.min, Math.min(...xs), 'min');
  eq(r.max, Math.max(...xs), 'max');
  close(r.mean, xs.reduce((a, b) => a + b, 0) / xs.length, 1e-9, 'mean');
  const round = JSON.parse(JSON.stringify(r));
  eq(round.count, xs.length, 'JSON.stringify sees count');
  return r;
}

check([3, 1, 4, 1, 5, 9, 2, 6]);
check([-2.5]);
check([0.1, 0.2, 0.3]);
check(Array.from({ length: 500 }, (_, i) => Math.cos(i) * 1e6));
const r = m.summarize(new Float64Array([]));
assert(r === null, `empty input must return null, got ${r === undefined ? 'undefined' : JSON.stringify(r)}`);

assert(/export function summarize\(\s*\w+: Float64Array\s*\):/.test(dts),
  `generated declaration does not take a Float64Array:\n${dts}`);
