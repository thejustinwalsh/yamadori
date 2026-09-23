import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { buildChunks, physicsStep, renderColor } from './solution.ts';

type _r = Expect<Equal<ReturnType<typeof buildChunks>, [string, string]>>;
export const _neg = () => {
  // @ts-expect-error -- takes no arguments
  buildChunks([physicsStep]);
};

const count = (s: string, re: RegExp) => (s.match(re) ?? []).length;
const [first, second] = buildChunks();

// Chunk 1 is complete on its own. Identifiers are read from the output rather
// than assumed, so any naming strategy that keeps them consistent passes.
const structName = /struct\s+(\w+)\s*\{\s*pos\s*:\s*vec3f\s*,\s*vel\s*:\s*vec3f/.exec(first)?.[1];
assert.ok(structName, 'first chunk declares the particle struct\n' + first);
const speedName = /fn\s+(\w+)\s*\(\s*p\s*:\s*\w+\s*\)\s*->\s*f32\s*\{\s*return\s+length/.exec(first)?.[1];
assert.ok(speedName, 'first chunk declares speed\n' + first);
const stepName = /fn\s+(\w+)\s*\(\s*p\s*:\s*\w+\s*,\s*dt\s*:\s*f32\s*\)/.exec(first)?.[1];
assert.ok(stepName, 'first chunk declares physicsStep\n' + first);
assert.equal(count(first, /\bfn\s/g), 2, 'first chunk declares speed and physicsStep only\n' + first);
assert.equal(count(first, /\bstruct\s/g), 1, 'first chunk declares one struct\n' + first);

// Chunk 2 only adds what is new, and reuses chunk 1's identifiers.
assert.equal(count(second, /\bstruct\s/g), 0, 'second chunk re-declares no struct\n' + second);
assert.equal(count(second, /\bfn\s/g), 1, 'second chunk declares only renderColor\n' + second);
const colorDecl = /fn\s+(\w+)\s*\(\s*p\s*:\s*(\w+)\s*\)\s*->\s*vec4f/.exec(second);
assert.ok(colorDecl, 'second chunk declares renderColor\n' + second);
assert.equal(colorDecl[2], structName, 'renderColor takes the same struct identifier\n' + second);
assert.ok(new RegExp(`\\b${speedName}\\s*\\(`).test(second), `renderColor calls ${speedName}\n${second}`);

// The concatenation declares everything exactly once.
const all = first + '\n' + second;
for (const decl of [`struct\\s+${structName}\\b`, `fn\\s+${speedName}\\b`, `fn\\s+${stepName}\\b`, `fn\\s+${colorDecl[1]}\\b`]) {
  assert.equal(count(all, new RegExp(decl, 'g')), 1, `${decl} declared once in the concatenation\n${all}`);
}

// Calls do not interfere with each other.
const again = buildChunks();
assert.deepEqual(again, [first, second], 'a second buildChunks() call returns the same chunks');
assert.ok(renderColor);
