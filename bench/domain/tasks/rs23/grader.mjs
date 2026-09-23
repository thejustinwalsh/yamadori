import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { assert, eq, throws } from './wasm_helpers.mjs';

const jsPath = process.argv[2];
const dts = readFileSync(jsPath.replace(/\.js$/, '.d.ts'), 'utf8');
const m = createRequire(import.meta.url)(jsPath);

const A = m.Accumulator;
assert(typeof A === 'function', `no exported class Accumulator (exports: ${Object.keys(m).join(', ')})`);

// Behaviour first.
let acc;
try {
  acc = new A(10);
  void acc.total; // a class without a wasm-bindgen constructor throws here
} catch (e) {
  assert(false, `new Accumulator(10) did not produce a usable object: ${e.message}`);
}
assert(typeof acc.total === 'number', `acc.total should be a number property, got ${typeof acc.total}`);
eq(acc.total, 10, 'initial total');
eq(acc.count, 0, 'initial count');
acc.add(2.5);
acc.add(-0.5);
eq(acc.total, 12, 'total after two adds');
eq(acc.count, 2, 'count after two adds');
eq(acc.add(1), undefined, 'add returns nothing');
acc.reset();
eq([acc.total, acc.count], [0, 0], 'after reset');
acc.add(4);
eq([acc.total, acc.count], [4, 1], 'add after reset');

// Read-only: assignment is rejected (module code is strict) and does not change the value.
try { acc.total = 99; } catch { /* TypeError: getter-only property */ }
try { acc.count = 99; } catch { /* TypeError: getter-only property */ }
eq([acc.total, acc.count], [4, 1], 'total/count cannot be assigned from JavaScript');

const b = new A(0);
b.add(1);
eq(acc.total, 4, 'instances are independent');
b.free();
throws(() => b.add(1), 'using an Accumulator after free()');
acc.free();

// Then the declaration.
const cls = dts.match(/export class Accumulator \{[\s\S]*?\n\}/);
assert(cls, `no 'export class Accumulator' in the declaration:\n${dts}`);
const body = cls[0];
for (const [re, what] of [[/\n\s*constructor\(\s*\w+: number\s*\);/, 'constructor(initial: number);'],
                          [/\n\s*add\(\s*\w+: number\s*\): void;/, 'add(x: number): void;'],
                          [/\n\s*reset\(\): void;/, 'reset(): void;'],
                          [/\n\s*readonly total: number;/, 'readonly total: number;'],
                          [/\n\s*readonly count: number;/, 'readonly count: number;'],
                          [/\n\s*free\(\): void;/, 'free(): void;']]) {
  assert(re.test(body), `declaration is missing '${what}':\n${body}`);
}
