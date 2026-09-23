import assert from 'node:assert/strict';
import tgpu, { type TgpuSlot } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect, IsAny } from './assert_types.ts';
import { cellOf, cellSize, gridSize } from './solution.ts';

type _slot = Expect<Equal<typeof gridSize, TgpuSlot<number>>>;
// cellSize's exact wrapper type is not prescribed; its behaviour is tested below.
type _nany = Expect<Equal<IsAny<typeof cellSize>, false>>;
export const _neg = () => {
  // @ts-expect-error -- gridSize holds a number
  const _s: TgpuSlot<string> = gridSize;
};

/** name -> value of every `const CELL_SIZE*: f32 = ...;` */
function consts(wgsl: string): Map<string, number> {
  const m = new Map<string, number>();
  for (const c of wgsl.matchAll(/const\s+(CELL_SIZE\w*)\s*:\s*f32\s*=\s*(?:f32\()?\s*([-\d.e]+)f?\s*\)?\s*;/g)) {
    m.set(c[1]!, Number.parseFloat(c[2]!));
  }
  return m;
}
/** fn name -> the CELL_SIZE* identifier its body uses */
function cellFns(wgsl: string): Map<string, string> {
  const m = new Map<string, string>();
  for (const f of wgsl.matchAll(/fn\s+(cellOf\w*)\s*\(\s*p\s*:\s*vec2f\s*\)\s*->\s*vec2u\s*\{([^}]*)\}/g)) {
    const used = /\b(CELL_SIZE\w*)\b/.exec(f[2]!);
    assert.ok(used, `${f[1]} must read the CELL_SIZE constant\n${wgsl}`);
    m.set(f[1]!, used[1]!);
  }
  return m;
}
const near = (a: number | undefined, b: number) => a !== undefined && Math.abs(a - b) < 1e-6;

// Default slot value.
const def = tgpu.resolve([cellOf]);
const defC = consts(def), defF = cellFns(def);
assert.equal(defF.size, 1, 'one cellOf\n' + def);
assert.ok(near(defC.get([...defF.values()][0]!), 1 / 16), 'CELL_SIZE = 1/16 by default\n' + def);

// Overridden.
const eight = tgpu.resolve([cellOf.with(gridSize, 8)]);
const eC = consts(eight), eF = cellFns(eight);
assert.ok(near(eC.get([...eF.values()][0]!), 1 / 8), 'CELL_SIZE = 1/8 with gridSize 8\n' + eight);
assert.ok(![...eC.values()].some((v) => near(v, 1 / 16)), 'the default value is not emitted\n' + eight);

// Two specialisations in one shader: each copy gets its own constant.
const main = tgpu.fn([], d.f32)`() -> f32 {
  let a = fine(vec2f(1.0));
  let b = coarse(vec2f(1.0));
  return 1.0;
}`.$uses({ fine: cellOf, coarse: cellOf.with(gridSize, 4) }).$name('main');
const both = tgpu.resolve([main]);
const bC = consts(both), bF = cellFns(both);
assert.equal(bF.size, 2, 'two cellOf variants\n' + both);
const values = [...bF.values()].map((c) => bC.get(c)).sort();
assert.ok(values.length === 2 && near(values[0], 1 / 16) && near(values[1], 1 / 4),
  `each variant reads its own CELL_SIZE (1/16 and 1/4): got ${values}\n${both}`);
