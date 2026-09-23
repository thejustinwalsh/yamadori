import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { diffTags, jaccard } from './solution.ts';

const before: ReadonlySet<string> = new Set(['ts', 'rust', 'wasm']);
const after: ReadonlySet<string> = new Set(['ts', 'wasm', 'gpu', 'tsl']);
const d = diffTags(before, after);
type _d = Expect<Equal<typeof d.added, Set<string>>>;
assert.deepEqual(d.added, new Set(['gpu', 'tsl']));
assert.deepEqual(d.removed, new Set(['rust']));
assert.deepEqual(d.kept, new Set(['ts', 'wasm']));
assert.deepEqual([...before], ['ts', 'rust', 'wasm'], 'inputs are not modified');
assert.deepEqual([...after], ['ts', 'wasm', 'gpu', 'tsl']);

// Results are fresh Sets, never the caller's objects.
const same = new Set(['a', 'b']);
const s = diffTags(same, same);
for (const out of [s.added, s.removed, s.kept]) {
  assert.ok(out instanceof Set);
  assert.notEqual(out, same);
}
s.kept.add('zzz');
assert.equal(same.has('zzz'), false);
assert.deepEqual(s.kept, new Set(['a', 'b', 'zzz']));

const e = diffTags(new Set(), new Set(['x']));
assert.deepEqual(e, { added: new Set(['x']), removed: new Set(), kept: new Set() });

assert.equal(jaccard(new Set(['a', 'b']), new Set(['b', 'c'])), 1 / 3);
assert.equal(jaccard(before, after), 2 / 5);
assert.equal(jaccard(new Set(['a']), new Set(['a'])), 1);
assert.equal(jaccard(new Set(['a']), new Set(['b'])), 0);
assert.equal(jaccard(new Set(), new Set()), 1, 'two empty sets are identical');
assert.equal(jaccard(new Set(), new Set(['a'])), 0);
