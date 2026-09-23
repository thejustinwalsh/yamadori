import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { get, type PathValue, type Paths } from './solution.ts';

type Cfg = {
  server: { host: string; port: number; tls: { cert: string; enabled: boolean } };
  tags: string[];
  created: Date;
  onReady: () => void;
  debug: boolean;
};

type _paths = Expect<Equal<Paths<Cfg>,
  | 'server' | 'server.host' | 'server.port' | 'server.tls' | 'server.tls.cert' | 'server.tls.enabled'
  | 'tags' | 'created' | 'onReady' | 'debug'>>;

interface Flat { a: number; b: string }
type _iface = Expect<Equal<Paths<Flat>, 'a' | 'b'>>;
type _deep = Expect<Equal<Paths<{ a: { b: { c: { d: 1 } } } }>, 'a' | 'a.b' | 'a.b.c' | 'a.b.c.d'>>;

type _v1 = Expect<Equal<PathValue<Cfg, 'server.tls'>, { cert: string; enabled: boolean }>>;
type _v2 = Expect<Equal<PathValue<Cfg, 'server.port'>, number>>;
type _v3 = Expect<Equal<PathValue<Cfg, 'tags'>, string[]>>;
type _v4 = Expect<Equal<PathValue<Cfg, 'server.tls.enabled'>, boolean>>;

const created = new Date(0);
const onReady = () => {};
const cfg: Cfg = {
  server: { host: 'localhost', port: 8080, tls: { cert: 'pem', enabled: true } },
  tags: ['a', 'b'],
  created,
  onReady,
  debug: false,
};

const port = get(cfg, 'server.port');
type _g1 = Expect<Equal<typeof port, number>>;
assert.equal(port, 8080);
const tls = get(cfg, 'server.tls');
type _g2 = Expect<Equal<typeof tls, { cert: string; enabled: boolean }>>;
assert.equal(tls, cfg.server.tls);
const cert = get(cfg, 'server.tls.cert');
type _g3 = Expect<Equal<typeof cert, string>>;
assert.equal(cert, 'pem');
assert.equal(get(cfg, 'debug'), false);
assert.equal(get(cfg, 'created'), created);
assert.equal(get(cfg, 'onReady'), onReady);
assert.deepEqual(get(cfg, 'tags'), ['a', 'b']);

function compileOnly(): void {
  // @ts-expect-error -- no such path
  get(cfg, 'server.nope');
  // @ts-expect-error -- arrays are leaves
  get(cfg, 'tags.length');
  // @ts-expect-error -- Dates are leaves
  get(cfg, 'created.getTime');
  // @ts-expect-error -- not a path at all
  get(cfg, '');
}
void compileOnly;
