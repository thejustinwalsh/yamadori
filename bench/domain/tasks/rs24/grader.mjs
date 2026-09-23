import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, eq } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
const m = createRequire(import.meta.url)(jsPath);

const f = m.parseHexColor;
assert(typeof f === 'function', `no export named parseHexColor (exports: ${Object.keys(m).join(', ')})`);

for (const [s, rgb] of [['#ff8000', [255, 128, 0]], ['#000000', [0, 0, 0]], ['#FfFfFf', [255, 255, 255]],
                        ['#1a2B3c', [0x1a, 0x2b, 0x3c]], ['#abc', [0xaa, 0xbb, 0xcc]], ['#F0a', [0xff, 0x00, 0xaa]],
                        ['#000', [0, 0, 0]]]) {
  let out;
  try { out = f(s); } catch (e) { assert(false, `parseHexColor(${JSON.stringify(s)}) threw: ${e?.message ?? e}`); }
  assert(out instanceof Uint8Array, `parseHexColor(${JSON.stringify(s)}) returns a Uint8Array, got ${out}`);
  eq(Array.from(out), rgb, `parseHexColor(${JSON.stringify(s)})`);
}

for (const bad of ['ff8000', '#ff800', '#ff80000', '#gg0000', '#12', '', '#', '# abc', '#abé', 'nope#abc']) {
  let threw = false, err;
  try { f(bad); } catch (e) { threw = true; err = e; }
  assert(threw, `parseHexColor(${JSON.stringify(bad)}) must throw`);
  assert(err instanceof Error,
    `parseHexColor(${JSON.stringify(bad)}) threw a ${typeof err} (${JSON.stringify(err)}), not an Error instance`);
  assert(typeof err.message === 'string' && err.message.includes(bad),
    `error message ${JSON.stringify(err.message)} does not contain the input ${JSON.stringify(bad)}`);
}

assert(/export function parseHexColor\(\s*\w+: string\s*\): Uint8Array;/.test(dts),
  `generated declaration is not parseHexColor(string): Uint8Array:\n${dts}`);
