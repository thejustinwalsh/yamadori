import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { assertNever, match } from './solution.ts';

type Shape =
  | { kind: 'circle'; r: number }
  | { kind: 'square'; side: number }
  | { kind: 'rect'; w: number; h: number };

const area = (s: Shape) =>
  match(s, {
    circle: (c) => Math.PI * c.r ** 2,
    square: (q) => q.side ** 2,
    rect: (r) => r.w * r.h,
  });
type _area = Expect<Equal<ReturnType<typeof area>, number>>;
assert.equal(area({ kind: 'square', side: 3 }), 9);
assert.equal(area({ kind: 'rect', w: 2, h: 5 }), 10);
assert.ok(Math.abs(area({ kind: 'circle', r: 1 }) - Math.PI) < 1e-12);

const seen: string[] = [];
const label = match({ kind: 'rect', w: 1, h: 2 } as Shape, {
  circle: (c) => { seen.push('circle'); type _c = Expect<Equal<typeof c, { kind: 'circle'; r: number }>>; return `c${c.r}`; },
  square: (q) => { seen.push('square'); type _q = Expect<Equal<typeof q, { kind: 'square'; side: number }>>; return `q${q.side}`; },
  rect: (r) => { seen.push('rect'); type _r = Expect<Equal<typeof r, { kind: 'rect'; w: number; h: number }>>; return `r${r.w}`; },
});
type _label = Expect<Equal<typeof label, string>>;

const mixed = (s: Shape) => match(s, { circle: () => 1, square: () => 'two', rect: () => true });
type _mixed = Expect<Equal<ReturnType<typeof mixed>, number | string | boolean>>;
assert.equal(mixed({ kind: 'square', side: 0 }), 'two');
assert.equal(label, 'r1');
assert.deepEqual(seen, ['rect'], 'only the matching handler runs');

const rect = { kind: 'rect', w: 1, h: 1 } as const;
assert.equal(match(rect as Shape, { circle: () => 0, square: () => 0, rect: (x) => x.w + 41 }), 42);

function describe(s: Shape): string {
  switch (s.kind) {
    case 'circle': return 'round';
    case 'square':
    case 'rect': return 'boxy';
    default: return assertNever(s);
  }
}
assert.equal(describe({ kind: 'circle', r: 2 }), 'round');
assert.throws(() => assertNever({ kind: 'hexagon' } as never), Error);

function compileOnly(s: Shape): void {
  // @ts-expect-error -- missing the rect handler
  match(s, { circle: () => 1, square: () => 2 });
  // @ts-expect-error -- circle has no `side`
  match(s, { circle: (c) => c.side, square: (q) => q.side, rect: (r) => r.w });
  // @ts-expect-error -- assertNever only accepts never
  assertNever(s);
  function partialSwitch(t: Shape): string {
    switch (t.kind) {
      case 'circle': return 'c';
      // @ts-expect-error -- 'square' and 'rect' are not handled, so t is not never here
      default: return assertNever(t);
    }
  }
  void partialSwitch;
}
void compileOnly;
