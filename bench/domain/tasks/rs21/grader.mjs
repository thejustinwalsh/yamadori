import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, eq } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
const m = createRequire(import.meta.url)(jsPath);

assert(typeof m.find_byte === 'function', `no export find_byte (exports: ${Object.keys(m).join(', ')})`);
const hay = new Uint8Array([7, 3, 9, 3, 0, 255]);
eq(m.find_byte(hay, 3, 0), 1, 'first 3');
eq(m.find_byte(hay, 3, 2), 3, 'next 3 from index 2');
eq(m.find_byte(hay, 3, 3), 3, 'from is inclusive');
eq(m.find_byte(hay, 7, 0), 0, 'match at index 0 (0 is not "absent")');
eq(m.find_byte(hay, 255, 0), 5, 'byte 255');
eq(m.find_byte(hay, 0, 1), 4, 'byte 0');

const absent = [[hay, 42, 0, 'needle not present'], [hay, 7, 1, 'only occurrence is before from'],
                [hay, 3, 6, 'from == length'], [hay, 3, 1000, 'from past the end'],
                [new Uint8Array([]), 0, 0, 'empty haystack']];
for (const [h, n, f, what] of absent) {
  const r = m.find_byte(h, n, f);
  assert(r === undefined, `${what}: expected undefined, got ${r === null ? 'null' : JSON.stringify(r)}`);
}
const big = new Uint8Array(100000);
big[99999] = 1;
eq(m.find_byte(big, 1, 5), 99999, 'large haystack');

// The declaration is part of the contract too.
assert(/export function find_byte\(\s*\w+: Uint8Array,\s*\w+: number,\s*\w+: number\s*\): number \| undefined;/.test(dts),
  `generated declaration is not find_byte(Uint8Array, number, number): number | undefined:\n${dts}`);
