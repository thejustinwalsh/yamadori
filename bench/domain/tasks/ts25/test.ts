import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { deepFreeze, type DeepReadonly } from './solution.ts';

type Config = {
  name: string;
  limits: { cpu: number; tags: string[] };
  pair: [string, { x: number }];
  owners: Map<string, { email: string }>;
  flags: Set<{ on: boolean }>;
  onChange: (v: number) => void;
  count?: number;
};

type _cfg = Expect<Equal<DeepReadonly<Config>, {
  readonly name: string;
  readonly limits: { readonly cpu: number; readonly tags: readonly string[] };
  readonly pair: readonly [string, { readonly x: number }];
  readonly owners: ReadonlyMap<string, { readonly email: string }>;
  readonly flags: ReadonlySet<{ readonly on: boolean }>;
  readonly onChange: (v: number) => void;
  readonly count?: number;
}>>;
type _prim = Expect<Equal<DeepReadonly<string>, string>>;
type _arr = Expect<Equal<DeepReadonly<{ a: number }[][]>, readonly (readonly { readonly a: number }[])[]>>;
type _fn = Expect<Equal<DeepReadonly<() => { a: 1 }>, () => { a: 1 }>>;

const onChange = (_v: number) => {};
const cfg: Config = {
  name: 'svc',
  limits: { cpu: 2, tags: ['a'] },
  pair: ['p', { x: 1 }],
  owners: new Map([['ann', { email: 'a@x' }]]),
  flags: new Set([{ on: true }]),
  onChange,
};
(cfg as Record<string, unknown>).loop = cfg;

const frozen = deepFreeze(cfg);
type _f = Expect<Equal<typeof frozen, DeepReadonly<Config>>>;
assert.equal(frozen, cfg as unknown, 'returns the same object');
assert.ok(Object.isFrozen(cfg));
assert.ok(Object.isFrozen(cfg.limits));
assert.ok(Object.isFrozen(cfg.limits.tags));
assert.ok(Object.isFrozen(cfg.pair));
assert.ok(Object.isFrozen(cfg.pair[1]));
assert.ok(Object.isFrozen(cfg.owners.get('ann')), 'Map values are frozen');
assert.ok(Object.isFrozen([...cfg.flags][0]), 'Set members are frozen');
assert.throws(() => { (cfg.limits.tags as string[]).push('b'); }, TypeError);
assert.throws(() => { (cfg.pair[1] as { x: number }).x = 2; }, TypeError);
assert.equal(frozen.onChange, onChange);
assert.equal(frozen.owners.get('ann')?.email, 'a@x');

assert.equal(deepFreeze(5), 5);
assert.equal(deepFreeze(null), null);

function compileOnly(): void {
  // @ts-expect-error -- nested arrays are readonly
  frozen.limits.tags.push('x');
  // @ts-expect-error -- nested objects are readonly
  frozen.limits.cpu = 3;
  // @ts-expect-error -- tuple members are readonly
  frozen.pair[1].x = 5;
  // @ts-expect-error -- maps are read-only
  frozen.owners.set('bob', { email: 'b@x' });
  // @ts-expect-error -- map values are readonly
  frozen.owners.get('ann')!.email = 'z';
  // @ts-expect-error -- sets are read-only
  frozen.flags.add({ on: false });
}
void compileOnly;
