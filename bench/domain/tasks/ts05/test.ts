import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { latestFor, leaderboard, withCorrection, type Score } from './solution.ts';

const freezeAll = (xs: Score[]): readonly Score[] => Object.freeze(xs.map((s) => Object.freeze(s)));
const scores = freezeAll([
  { player: 'ann', points: 10, at: 1 },
  { player: 'bob', points: 30, at: 2 },
  { player: 'ann', points: 30, at: 3 },
  { player: 'cy', points: 5, at: 4 },
  { player: 'bob', points: 10, at: 5 },
]);
const before = JSON.stringify(scores);

const top = leaderboard(scores);
type _top = Expect<Equal<typeof top, Score[]>>;
assert.deepEqual(top.map((s) => `${s.player}@${s.at}`), ['bob@2', 'ann@3', 'ann@1', 'bob@5', 'cy@4']);
assert.notEqual(top, scores as unknown);
top.push({ player: 'x', points: 0, at: 9 }); // the result is a fresh mutable array
assert.equal(scores.length, 5);

const last = latestFor(scores, 'ann');
type _last = Expect<Equal<typeof last, Score | undefined>>;
assert.equal(last, scores[2]);
assert.equal(latestFor(scores, 'bob'), scores[4]);
assert.equal(latestFor(scores, 'nobody'), undefined);

const fixed = withCorrection(scores, 1, 99);
assert.deepEqual(fixed[1], { player: 'bob', points: 99, at: 2 });
assert.equal(fixed[0], scores[0]);
assert.equal(fixed.length, 5);
const fixedLast = withCorrection(scores, -1, 7);
assert.deepEqual(fixedLast[4], { player: 'bob', points: 7, at: 5 });
assert.deepEqual(fixedLast.slice(0, 4), scores.slice(0, 4));
assert.throws(() => withCorrection(scores, 5, 1), RangeError);
assert.throws(() => withCorrection(scores, -6, 1), RangeError);

assert.equal(JSON.stringify(scores), before, 'input must not be mutated');
assert.deepEqual(leaderboard([]), []);
