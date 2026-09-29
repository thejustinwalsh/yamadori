'use strict';
// yama_resolve_packages's DRIVER (mcp/npm_resolve.py): npm's OWN resolver,
// run inside the pinned PackageLens image (node 24.21.0, npm 11.19.0 -- the
// base image's bundled npm; no new download) as `node -`, this file on stdin
// and the request in YAMA_RESOLVE_INPUT (JSON). Nothing here executes package
// code: every install is `--package-lock-only --ignore-scripts` (no tarball
// is unpacked, no node_modules, no lifecycle script), in a fresh directory
// under /tmp (the container's only writable place, a tmpfs gone with it).
// Its network is the container's: the egress gate only (mcp_host.run_argv).
//
// What it does, all mechanically from the registry's package documents:
//   1. each spec parsed by npm's own npm-package-arg: registry names, ranges,
//      versions and dist-tags only (a git URL, a path or an alias is refused);
//   2. each name's abbreviated packument (the document npm itself installs
//      from) read once; a range no STABLE version satisfies takes a
//      prerelease only when the caller allowed it (allow_prerelease) or named
//      one -- npm semver's default otherwise;
//   3. `npm install --package-lock-only` of the set, then of the EXACT set it
//      chose (npm 11 accepted a root range whose chosen version sits outside
//      a peer range, and refused the same versions pinned exactly). On
//      ERESOLVE in either, npm's own conflict report is parsed and ONE
//      package is moved within what the caller asked for -- the peer to the
//      newest version every fixed peer range on it admits, else the
//      dependent to the newest version whose own peer range admits the peer
//      -- and npm runs again; a package is moved at most once, so the
//      attempts are bounded by the number of names;
//   4. from package-lock.json: each requested package's exact version, every
//      peer npm installed for them (lockfile `peer: true`) with the ranges
//      the set declares on it, and the optional peers npm left out;
//   5. type definitions: for a requested package or peer that ships none (no
//      `types` / `typings`, no `types` condition in `exports`), the
//      DefinitelyTyped package @types/<name> whose major.minor matches (the
//      DefinitelyTyped convention: "the major and minor version of the
//      @types package match the library's"), else the same major's newest
//      lower minor (a major 0 library's minor is its major: npm's caret rule);
//   6. the type packages added as dev dependencies to the exact set npm just
//      resolved, so both install lines are npm's verdict, not ours.
// Prints ONE line: `YAMA_RESOLVE_RESULT <json>`.
//
// The offline suite (mcp/test_npm_resolve.py) runs this file with the host's
// node against a fake npm (YAMA_RESOLVE_NPM_CLI), a local registry and a temp
// work directory (YAMA_RESOLVE_WORK); the container sets none of them.

const cp = require('child_process');
const fs = require('fs');
const path = require('path');

// npm's own directory: <prefix>/lib/node_modules/npm (the Linux image), or
// <node dir>/node_modules/npm (Windows' installer).
const NPM = [path.join(path.dirname(process.execPath), '..', 'lib', 'node_modules', 'npm'),
             path.join(path.dirname(process.execPath), 'node_modules', 'npm')]
  .find(d => fs.existsSync(path.join(d, 'package.json')));
const NPM_CLI = process.env.YAMA_RESOLVE_NPM_CLI || path.join(NPM, 'bin', 'npm-cli.js');
const npa = require(path.join(NPM, 'node_modules', 'npm-package-arg'));
const semver = require(path.join(NPM, 'node_modules', 'semver'));

const IN = JSON.parse(process.env.YAMA_RESOLVE_INPUT || '{}');
const REG = String(IN.registry || 'https://registry.npmjs.org').replace(/\/+$/, '');
const FETCH_TIMEOUT_MS = Number(IN.fetch_timeout_ms) || 300000;
const ALLOW_PRE = IN.allow_prerelease === true;
// npm's own Accept for the abbreviated packument (npm-registry-fetch).
const CORGI = 'application/vnd.npm.install-v1+json; q=1.0, application/json; q=0.8, */*';
const WORK = process.env.YAMA_RESOLVE_WORK || '/tmp/yama-resolve';

const out = {ok: false, npm_version: null, requested: [], packages: [], optional_peers: [],
             types: [], types_missing: [], fixes: [], conflicts: [], errors: [], runs: [],
             install: null, install_dev: null, lock_packages: 0,
             verified: false};

function done() {
  process.stdout.write('YAMA_RESOLVE_RESULT ' + JSON.stringify(out) + '\n');
}

const _pk = new Map();
async function packument(name) {
  if (_pk.has(name)) return _pk.get(name);
  const p = (async () => {
    const url = REG + '/' + name.replace('/', '%2F');
    const r = await fetch(url, {headers: {accept: CORGI},
                                signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)});
    if (r.status === 404) return null;
    if (!r.ok) throw new Error(`GET ${url} answered HTTP ${r.status}`);
    return await r.json();
  })();
  _pk.set(name, p);
  return p;
}

async function versionDoc(name, version) {
  const url = REG + '/' + name.replace('/', '%2F') + '/' + encodeURIComponent(version);
  const r = await fetch(url, {signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)});
  if (!r.ok) throw new Error(`GET ${url} answered HTTP ${r.status}`);
  return await r.json();
}

function npmInstall(dir, specs, dev) {
  const args = [NPM_CLI, 'install', '--package-lock-only', '--ignore-scripts', '--no-audit',
                '--no-fund', '--json'].concat(dev ? ['--save-dev'] : [], specs);
  const t0 = Date.now();
  const r = cp.spawnSync(process.execPath, args, {cwd: dir, encoding: 'utf8',
                                                  maxBuffer: 1 << 26});
  let json = null;
  try { json = JSON.parse(r.stdout); } catch (e) { json = null; }
  const run = {specs, dev: !!dev, exit: r.status, ms: Date.now() - t0};
  const err = json && json.error;
  if (err) run.error = {code: err.code, summary: String(err.summary || '').slice(0, 400),
                        detail: String(err.detail || '').slice(0, 4000)};
  else if (r.status !== 0) run.error = {code: 'NPM_FAILED',
                                        summary: String(r.stderr || '').slice(-400), detail: ''};
  out.runs.push(run);
  return run;
}

function freshDir() {
  fs.rmSync(WORK, {recursive: true, force: true});
  fs.mkdirSync(WORK, {recursive: true});
  // What `npm init -y` writes, less its prose: a name and a version.
  fs.writeFileSync(path.join(WORK, 'package.json'),
                   JSON.stringify({name: 'yama-resolve', version: '0.0.0', private: true}));
  return WORK;
}

function splitNameVersion(s) {
  const at = s.lastIndexOf('@');
  if (at <= 0) return {name: s, version: null};
  return {name: s.slice(0, at), version: s.slice(at + 1)};
}

// npm's ERESOLVE report (arborist's explain-eresolve), as npm 11 writes it:
//   Found: react@19.3.0 ... Could not resolve dependency:
//   peer react@">=19.0 <19.3" from @react-three/fiber@10.0.0-alpha.5
//   [Conflicting peer dependency: react@18.3.1]
function parseEresolve(detail) {
  const c = {};
  let m = /^Found: (\S+)/m.exec(detail);
  if (m) c.found = splitNameVersion(m[1]);
  m = /Could not resolve dependency:\n(?:(peer|peerOptional|dev|optional|prod) )?(\S+?)@"([^"]*)" from (the root project|\S+)/.exec(detail);
  if (m) {
    c.edge = {type: m[1] || 'prod', name: m[2], range: m[3]};
    c.from = m[4] === 'the root project' ? {name: null, version: null, root: true}
      : splitNameVersion(m[4]);
  }
  m = /Conflicting peer dependency: (\S+)/.exec(detail);
  if (m) c.conflicting = splitNameVersion(m[1]);
  return c;
}

function isPre(v) { return !!semver.prerelease(v); }

// The version npm-pick-manifest would reach for, among `cands`: a version not
// deprecated first; a stable one before a prerelease; among prereleases, a
// version a dist-tag names (the publisher's pointer, never an arbitrary
// per-commit build) before any other; the newest by npm's semver order.
function prefer(pk, cands) {
  if (!cands.length) return null;
  const vers = pk.versions || {};
  const live = cands.filter(v => !(vers[v] && vers[v].deprecated));
  const pool = live.length ? live : cands;
  const stable = pool.filter(v => !isPre(v));
  if (stable.length) return semver.rsort(stable.slice())[0];
  const heads = new Set(Object.values(pk['dist-tags'] || {}));
  const tagged = pool.filter(v => heads.has(v));
  return semver.rsort((tagged.length ? tagged : pool).slice())[0];
}

function channels(pk, range, except) {
  const tags = pk['dist-tags'] || {};
  return Object.entries(tags)
    .filter(([t, v]) => v !== except && semver.satisfies(v, range, {includePrerelease: true}))
    .map(([t, v]) => ({tag: t, version: v}))
    .sort((a, b) => semver.rcompare(a.version, b.version));
}

// Does version `v` satisfy what the CALLER asked (`range`)? `pre`: a
// prerelease counts (allow_prerelease, or a prerelease the caller named).
function satisfies(v, range, pre) {
  try { return semver.satisfies(v, range, {includePrerelease: !!pre, loose: true}); }
  catch (e) { return false; }
}

// Does version `v` satisfy a package's dependency or peer range, the way npm
// judges it (arborist lib/dep-valid.js: '*' and '' always; a range or version
// by semver.satisfies(version, range, loose) -- no includePrerelease, so a
// prerelease satisfies only a range that names a prerelease of its own
// major.minor.patch); a dist-tag is taken as satisfied, as arborist does.
function depValid(v, range) {
  const r = String(range == null ? '' : range).trim();
  if (r === '' || r === '*') return true;
  if (!semver.validRange(r, true)) return true;
  return semver.satisfies(v, r, true);
}

function typesName(name) {
  return '@types/' + (name.startsWith('@') ? name.slice(1).replace('/', '__') : name);
}

function hasTypesCondition(x) {
  if (!x || typeof x !== 'object') return false;
  if (Array.isArray(x)) return x.some(hasTypesCondition);
  return Object.entries(x).some(([k, v]) => k === 'types' || hasTypesCondition(v));
}

async function main() {
  // npm's version, from its own package.json.
  out.npm_version = require(path.join(NPM, 'package.json')).version;
  const reqs = [];
  const seen = new Set();
  for (const raw of IN.packages || []) {
    let p;
    try { p = npa(String(raw)); } catch (e) {
      out.errors.push({spec: raw, code: 'NOT_A_PACKAGE_NAME', why: String(e.message).slice(0, 200)});
      continue;
    }
    if (!p.registry || !['range', 'version', 'tag'].includes(p.type)) {
      out.errors.push({spec: raw, code: 'NOT_A_REGISTRY_SPEC',
                       why: `a ${p.type} spec, not a registry name with a version, range or dist-tag`});
      continue;
    }
    if (seen.has(p.name)) {
      out.errors.push({spec: raw, code: 'DUPLICATE', why: `${p.name} is asked for twice; the first is used`});
      continue;
    }
    seen.add(p.name);
    reqs.push({spec: String(raw), name: p.name, type: p.type, fetchSpec: p.fetchSpec,
               bare: p.type === 'range' && (p.rawSpec === '*' || p.rawSpec === '')});
  }
  // 1. every requested name's packument, together.
  const docs = await Promise.all(reqs.map(r => packument(r.name).then(
    d => ({d}), e => ({e: String(e.message || e).slice(0, 300)}))));
  const ok = [];
  reqs.forEach((r, i) => {
    const {d, e} = docs[i];
    if (e) { out.errors.push({spec: r.spec, code: 'REGISTRY_UNREACHABLE', why: e}); return; }
    if (!d) { out.errors.push({spec: r.spec, code: 'NOT_FOUND', why: `the npm registry has no package named ${r.name}`}); return; }
    r.pk = d;
    ok.push(r);
  });
  // 2. what each asks for, and the spec npm is given.
  const usable = [];
  for (const r of ok) {
    const vers = Object.keys(r.pk.versions || {});
    const tags = r.pk['dist-tags'] || {};
    r.channels = [];
    if (r.type === 'tag') {
      if (!tags[r.fetchSpec]) {
        out.errors.push({spec: r.spec, code: 'NO_SUCH_TAG',
                         why: `${r.name} has no dist-tag ${r.fetchSpec}; its dist-tags: ` +
                              Object.entries(tags).map(([t, v]) => `${t} ${v}`).join(', ')});
        continue;
      }
      r.want = tags[r.fetchSpec]; r.range = r.want; r.how = `dist-tag ${r.fetchSpec}`;
      r.install = `${r.name}@${r.fetchSpec}`;
    } else if (r.type === 'version') {
      if (!r.pk.versions[r.fetchSpec]) {
        out.errors.push({spec: r.spec, code: 'NO_SUCH_VERSION',
                         why: `${r.name} has no version ${r.fetchSpec}; its dist-tags: ` +
                              Object.entries(tags).map(([t, v]) => `${t} ${v}`).join(', ')});
        continue;
      }
      r.want = r.fetchSpec; r.range = r.fetchSpec; r.how = 'the version asked for';
      r.install = `${r.name}@${r.fetchSpec}`;
    } else {
      r.range = r.fetchSpec;
      const npmPick = semver.maxSatisfying(vers, r.range, {loose: true});
      if (npmPick) {
        r.how = r.bare ? 'any version' : `the range ${r.range}`;
        r.install = r.bare ? r.name : `${r.name}@${r.range}`;
      } else {
        const pre = vers.filter(v => satisfies(v, r.range, true));
        r.channels = channels(r.pk, r.range, null);
        if (!pre.length) {
          out.errors.push({spec: r.spec, code: 'NO_MATCHING_VERSION',
                           why: `no version of ${r.name} satisfies ${r.range}; its dist-tags: ` +
                                Object.entries(tags).map(([t, v]) => `${t} ${v}`).join(', ')});
          continue;
        }
        if (!ALLOW_PRE) {
          out.errors.push({spec: r.spec, code: 'ONLY_PRERELEASES',
                           why: `no stable version of ${r.name} satisfies ${r.range}; ` +
                                `prereleases do: ` + r.channels.map(c => `${c.tag} ${c.version}`).join(', '),
                           channels: r.channels});
          continue;
        }
        r.want = prefer(r.pk, pre);
        r.pre = true;
        const tag = Object.entries(tags).find(([t, v]) => v === r.want);
        r.how = `a prerelease in ${r.range}` + (tag ? ` (dist-tag ${tag[0]})` : '');
        r.channels = channels(r.pk, r.range, r.want);
        r.install = `${r.name}@${r.want}`;
      }
    }
    usable.push(r);
  }
  if (!usable.length) return;

  // 3. npm's resolver on the caller's specs, then on the exact set it chose
  // (npm 11 accepted `@react-three/fiber@alpha react@^19` with react 19.3.0
  // outside fiber's peer range `>=19.0 <19.3`, and refused the same versions
  // pinned exactly: 2026-09-29, this driver's live run). On ERESOLVE in
  // either, move one package and start again.
  const byName = new Map(usable.map(r => [r.name, r]));
  const pinned = new Map();          // name -> version a fix set
  const extra = [];                  // names a fix added that nobody asked for
  let st = null;
  for (;;) {
    const specs = usable.map(r => pinned.has(r.name) ? `${r.name}@${pinned.get(r.name)}` : r.install)
      .concat(extra.map(n => `${n}@${pinned.get(n)}`));
    let run = npmInstall(freshDir(), specs, false);
    if (!run.error) {
      st = readLock(byName, extra);
      run = npmInstall(freshDir(), st.core, false);
      if (!run.error) break;
    }
    if (run.error.code !== 'ERESOLVE') {
      out.errors.push({spec: run.specs.join(' '), code: run.error.code,
                       why: run.error.summary || run.error.detail.slice(0, 400)});
      return;
    }
    const c = parseEresolve(run.error.detail);
    const fix = c.edge && c.found ? await moveOne(c, byName, pinned) : null;
    if (!fix) {
      out.conflicts.push(await explain(c, byName, run.error));
      return;
    }
    out.fixes.push(fix);
    pinned.set(fix.name, fix.to);
    if (!byName.has(fix.name) && !extra.includes(fix.name)) extra.push(fix.name);
  }
  out.lock_packages = st.lock_packages;
  out.requested = st.requested;
  out.packages = st.peers;
  const {P, topNames, neededBy} = st;
  const top = n => P['node_modules/' + n];

  // 4. the optional peers of what was asked for that npm did not install
  // (npm installs a peer marked optional in peerDependenciesMeta only when
  // something else asks for it): each with every range the set declares on
  // it, so a caller that uses one names it in the next call.
  const installed = new Set(topNames);
  const optNames = new Set();
  for (const r of usable) {
    const e = top(r.name) || {};
    for (const [n, m] of Object.entries(e.peerDependenciesMeta || {})) {
      if (m && m.optional && !installed.has(n) && (e.peerDependencies || {})[n] !== undefined
          && !n.startsWith('@types/')) optNames.add(n);
    }
  }
  for (const n of optNames) out.optional_peers.push({name: n, needed_by: neededBy(n)});

  // 5. type definitions.
  const typed = out.requested.filter(r => r.version).map(r => [r.name, r.version])
    .concat(out.packages.map(p => [p.name, p.version]))
    .filter(([n]) => !n.startsWith('@types/'));
  const own = await Promise.all(typed.map(([n, v]) => versionDoc(n, v).then(
    d => !!(d.types || d.typings || hasTypesCondition(d.exports)), () => null)));
  for (let i = 0; i < typed.length; i++) {
    const [n, v] = typed[i];
    if (own[i] !== false) continue;
    const tn = typesName(n);
    if (byName.has(tn)) continue;
    let pk;
    try { pk = await packument(tn); } catch (e) { pk = null; }
    if (!pk) { out.types_missing.push({for: n, version: v, why: `the registry has no ${tn}`}); continue; }
    const M = semver.major(v), m = semver.minor(v);
    const vs = Object.keys(pk.versions || {}).filter(x => !isPre(x) && !(pk.versions[x] || {}).deprecated);
    let pick = semver.maxSatisfying(vs, `~${M}.${m}.0`), match = 'major.minor';
    if (!pick && M > 0) { pick = semver.maxSatisfying(vs, `>=${M}.0.0 <${M}.${m}.0`); match = 'major'; }
    if (!pick) {
      out.types_missing.push({for: n, version: v, why: `${tn} has no version for ${M}.${m}`});
      continue;
    }
    out.types.push({for: n, for_version: v, name: tn, version: pick, match});
  }
  // A type package the tree installs as a peer (e.g. @types/react for
  // @types/react-reconciler) that no entry above covers joins the dev line.
  for (const p of st.type_peers) {
    if (!out.types.some(t => t.name === p.name) && !byName.has(p.name)) {
      out.types.push({for: null, for_version: null, name: p.name, version: p.version,
                      match: 'peer', needed_by: p.needed_by});
    }
  }

  // 6. the type packages as dev dependencies, added to the set npm just
  // resolved exactly (WORK holds that lockfile), so the dev line is npm's
  // verdict beside the dependencies.
  const deps = st.core;
  const dev = out.types.map(t => `${t.name}@${t.version}`);
  if (dev.length) {
    const v2 = npmInstall(WORK, dev, true);
    if (v2.error) {
      out.types_missing.push(...out.types.map(t => ({for: t.for || t.name, version: t.for_version,
        why: `npm could not add ${t.name}@${t.version} beside the set (${v2.error.code})`})));
      out.types = [];
    } else {
      const L = JSON.parse(fs.readFileSync(path.join(WORK, 'package-lock.json'), 'utf8')).packages || {};
      for (const t of out.types) {
        const e = L['node_modules/' + t.name];
        if (e) t.version = e.version;
      }
    }
  }
  out.verified = true;
  out.install = 'npm install --save-exact ' + deps.join(' ');
  if (out.types.length) out.install_dev = 'npm install --save-exact --save-dev ' +
    out.types.map(t => `${t.name}@${t.version}`).join(' ');
  out.ok = true;
}

// What package-lock.json says: each requested package's version, the peers
// npm installed for the set (lockfile `peer: true`, needed as a non-optional
// peer by some installed package; a package that is only a DEPENDENCY of a
// peer, like csstype of @types/react, is npm's to install and not listed),
// the type packages among them, and the exact core set.
function readLock(byName, extra) {
  const lock = JSON.parse(fs.readFileSync(path.join(WORK, 'package-lock.json'), 'utf8'));
  const P = lock.packages || {};
  const topNames = Object.keys(P).filter(k => k.startsWith('node_modules/') &&
                                             k.split('node_modules/').length === 2)
    .map(k => k.slice('node_modules/'.length));
  const neededBy = name => {
    const by = [];
    for (const n of topNames) {
      const e = P['node_modules/' + n];
      const r = (e.peerDependencies || {})[name];
      if (r === undefined) continue;
      const opt = !!((e.peerDependenciesMeta || {})[name] || {}).optional;
      by.push({name: n, version: e.version, range: r, optional: opt});
    }
    return by;
  };
  const requested = [];
  for (const r of byName.values()) {
    const e = P['node_modules/' + r.name];
    requested.push({spec: r.spec, name: r.name, version: e ? e.version : null,
                    how: r.how, prerelease: !!(e && isPre(e.version)),
                    channels: r.channels, needed_by: neededBy(r.name),
                    fixed: out.fixes.some(f => f.name === r.name)});
  }
  const peers = [], typePeers = [];
  for (const n of topNames) {
    if (byName.has(n)) continue;
    const e = P['node_modules/' + n];
    const by = neededBy(n).filter(b => !b.optional);
    if (!(e.peer && by.length) && !extra.includes(n)) continue;
    (n.startsWith('@types/') ? typePeers : peers).push({name: n, version: e.version, needed_by: by});
  }
  const core = requested.filter(r => r.version).map(r => `${r.name}@${r.version}`)
    .concat(peers.map(p => `${p.name}@${p.version}`));
  return {P, topNames, neededBy, requested, peers, type_peers: typePeers, core,
          lock_packages: Object.keys(P).filter(k => k !== '').length};
}

// Move ONE package within what the caller asked for (see the header, 3).
async function moveOne(c, byName, pinned) {
  const Y = c.edge.name, peerRange = c.edge.range;
  const found = c.found && c.found.name === Y ? c.found : (c.conflicting || c.found);
  const yv = found && found.name === Y ? found.version : null;
  const D = c.from && !c.from.root ? c.from : null;
  // (a) the peer, to the newest version the dependent's range admits and the
  // caller's own spec for it allows.
  if (!pinned.has(Y)) {
    const ry = byName.get(Y);
    const exact = ry && (ry.type === 'version' || ry.type === 'tag');
    if (!exact) {
      let pk;
      try { pk = await packument(Y); } catch (e) { pk = null; }
      if (pk) {
        const allowPre = ALLOW_PRE || (ry && ry.pre);
        // Every peer range on Y that a package of the set already fixed at
        // one version declares (a version or dist-tag asked for, a
        // prerelease picked, a package an earlier fix moved).
        const ranges = [peerRange];
        for (const r of byName.values()) {
          const at = pinned.get(r.name) || r.want;
          const e = at && r.pk && (r.pk.versions || {})[at];
          const pr = e && (e.peerDependencies || {})[Y];
          if (pr !== undefined && r.name !== Y) ranges.push(pr);
        }
        const cands = Object.keys(pk.versions || {}).filter(v =>
          ranges.every(pr => depValid(v, pr)) &&
          (!ry || ry.bare || satisfies(v, ry.range, allowPre && isPre(v))) &&
          (allowPre || !isPre(v)));
        const to = prefer(pk, cands);
        if (to && to !== yv) {
          return {name: Y, from: yv, to, because: D ? `${D.name}@${D.version} needs ${Y} ${peerRange}`
                                                    : `the tree needs ${Y} ${peerRange}`,
                  asked: ry ? ry.spec : null};
        }
      }
    }
  }
  // (b) the dependent, to the newest version whose own peer range admits the
  // peer as it stands, within the caller's spec for it.
  if (D && yv && !pinned.has(D.name) && byName.has(D.name)) {
    const rd = byName.get(D.name);
    if (rd.type === 'range') {
      let pk;
      try { pk = await packument(D.name); } catch (e) { pk = null; }
      if (pk) {
        const allowPre = ALLOW_PRE || rd.pre;
        const cands = Object.keys(pk.versions || {}).filter(v => {
          if (!satisfies(v, rd.range, allowPre && isPre(v))) return false;
          if (!allowPre && isPre(v)) return false;
          const pr = ((pk.versions[v] || {}).peerDependencies || {})[Y];
          // A version that does not declare the peer at all is an older
          // design (drei 3.x peers on react-three-fiber, not
          // @react-three/fiber), not a fit.
          return pr !== undefined && depValid(yv, pr);
        });
        const to = prefer(pk, cands);
        if (to && to !== D.version) {
          return {name: D.name, from: D.version, to,
                  because: `${Y} ${yv} is outside ${D.name}@${D.version}'s peer range ${peerRange}`,
                  asked: rd.spec};
        }
      }
    }
  }
  return null;
}

// A conflict nothing within the caller's specs resolves: npm's words, and the
// newest version of each side that would fit the other, ignoring the specs.
async function explain(c, byName, err) {
  const x = {summary: err.summary, found: c.found || null, edge: c.edge || null,
             from: c.from || null, conflicting: c.conflicting || null};
  try {
    if (c.edge) {
      const pk = await packument(c.edge.name);
      if (pk) x.peer_fits = prefer(pk, Object.keys(pk.versions || {}).filter(v =>
        depValid(v, c.edge.range) && (ALLOW_PRE || !isPre(v))));
      const yv = c.found && c.found.name === c.edge.name ? c.found.version : null;
      if (c.from && !c.from.root && yv) {
        const pd = await packument(c.from.name);
        if (pd) x.dependent_fits = prefer(pd, Object.keys(pd.versions || {}).filter(v => {
          if (!ALLOW_PRE && isPre(v)) return false;
          const pr = ((pd.versions[v] || {}).peerDependencies || {})[c.edge.name];
          return pr !== undefined && depValid(yv, pr);
        }));
      }
    }
  } catch (e) { x.why = String(e.message || e).slice(0, 200); }
  x.asked = [c.edge && c.edge.name, c.from && c.from.name].filter(n => n && byName.has(n))
    .map(n => byName.get(n).spec);
  return x;
}

main().catch(e => {
  out.errors.push({spec: null, code: 'DRIVER_FAILED', why: String(e && e.stack || e).slice(0, 600)});
}).finally(done);
