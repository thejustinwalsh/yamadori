// Package onboarding's pure helpers (api/onboarding.ts), its route, and
// static renders of the stage rail and the blocker list (vitest runs in
// node: react-dom/server).
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  canForce,
  clarifyNeeds,
  curveRows,
  erroredJobOf,
  evalView,
  latestJob,
  linksOf,
  lostNames,
  onboardingBody,
  packageLabel,
  railCells,
  rerunnableJobs,
  scalars,
  splitList,
  tier3Pending,
  waitingLine,
  type Blocker,
  type OnboardingDetail,
  type OnboardingJob,
} from '../api/onboarding';
import { href, resolve } from '../router';
import { Blockers, OnboardingRail } from './skills/onboarding';

const STAGES = ['submitted', 'resolve', 'clarify', 'index', 'vocab', 'examples', 'knn', 'sources', 'skills', 'retire', 'rebuild', 'evaluate', 'complete'];
const JOINS = ['skills', 'rebuild'];
const NOW = 1_800_000_000;

const job = (over: Partial<OnboardingJob>): OnboardingJob => ({
  id: 'j1',
  queue: 'package.resolve',
  lane: 'net',
  state: 'done',
  stage: 'resolve',
  attempts: 1,
  max_attempts: 3,
  error: null,
  progress: null,
  created: 1,
  started: 1,
  finished: 2,
  worker: 'w',
  not_before: null,
  parent: null,
  ...over,
});

type RailInput = Pick<OnboardingDetail, 'stages' | 'joins' | 'stage' | 'job_rows' | 'blockers' | 'held'>;
const at = (stage: string, job_rows: OnboardingJob[] = [], blockers: Blocker[] = [], held: string | null = null): RailInput => ({ stages: STAGES, joins: JOINS, stage, job_rows, blockers, held });
const states = (d: RailInput) => Object.fromEntries(railCells(d, NOW).map((c) => [c.stage, c.state]));

describe('the create body (PROMPT + LINKS)', () => {
  const f = { prompt: '', links: '', aliases: '', replaces: 'auto' as const };
  it('reads links in the prompt and the field, once each, trailing punctuation off', () => {
    expect(linksOf('Add koota: https://github.com/pmndrs/koota.', ['https://www.npmjs.com/package/koota', 'https://github.com/pmndrs/koota'])).toEqual(['https://github.com/pmndrs/koota', 'https://www.npmjs.com/package/koota']);
    expect(linksOf('no link here', [])).toEqual([]);
  });
  it('splits lists on commas and newlines, links also on spaces', () => {
    expect(splitList('r3f, fiber\nthree fiber')).toEqual(['r3f', 'fiber', 'three fiber']);
    expect(splitList('https://a.dev/x https://b.dev/y,https://c.dev', true)).toEqual(['https://a.dev/x', 'https://b.dev/y', 'https://c.dev']);
  });
  it('sends the prompt alone when that is all there is, and leaves replaces to the rule on auto', () => {
    expect(onboardingBody({ ...f, prompt: ' Add koota 0.6.6 https://github.com/pmndrs/koota ' })).toEqual({ body: { prompt: 'Add koota 0.6.6 https://github.com/pmndrs/koota' } });
  });
  it('sends links, aliases (bound or not) and an explicit replace/alongside', () => {
    expect(onboardingBody({ prompt: 'Add r3f and drei', links: 'https://www.npmjs.com/package/@react-three/fiber\nhttps://www.npmjs.com/package/@react-three/drei', aliases: 'r3f=@react-three/fiber, drei', replaces: 'alongside' })).toEqual({
      body: { prompt: 'Add r3f and drei', links: ['https://www.npmjs.com/package/@react-three/fiber', 'https://www.npmjs.com/package/@react-three/drei'], aliases: ['r3f=@react-three/fiber', 'drei'], replaces: 'alongside' },
    });
  });
  it('refuses what the server would refuse, saying why', () => {
    expect(onboardingBody({ ...f, prompt: '   ' })).toHaveProperty('error');
    expect(onboardingBody({ ...f, prompt: 'Add koota' })).toHaveProperty('error');
    expect(onboardingBody({ ...f, prompt: 'Add koota', links: 'github.com/pmndrs/koota' })).toHaveProperty('error');
    expect(onboardingBody({ ...f, prompt: 'Add koota https://github.com/pmndrs/koota', aliases: '=koota' })).toHaveProperty('error');
  });
});

describe('the stage rail', () => {
  it('draws done stages before where the dataset stands and pending after', () => {
    const st = states(at('index', [job({}), job({ id: 'j2', queue: 'package.index', lane: 'gpu', stage: 'index', state: 'running', created: 3 })]));
    expect(st.submitted).toBe('done');
    expect(st.resolve).toBe('done');
    expect(st.clarify).toBe('done');
    expect(st.index).toBe('running');
    expect(st.vocab).toBe('pending');
    expect(st.skills).toBe('pending');
    expect(st.complete).toBe('pending');
  });
  it('shows a gpu stage deferred for an idle stack as waiting, and one merely queued as queued', () => {
    const deferred = job({ id: 'j2', stage: 'index', state: 'queued', not_before: NOW + 60, created: 3 });
    expect(states(at('index', [deferred])).index).toBe('waiting');
    expect(states(at('index', [{ ...deferred, not_before: NOW - 1 }])).index).toBe('queued');
  });
  it('takes the newest job of a stage: a re-run that succeeded is done', () => {
    const rows = [job({ id: 'old', stage: 'sources', state: 'errored', created: 5 }), job({ id: 'new', stage: 'sources', state: 'done', created: 9 })];
    expect(latestJob(rows, 'sources')?.id).toBe('new');
    expect(states(at('skills', rows)).sources).toBe('done');
    expect(states(at('sources', [job({ stage: 'sources', state: 'errored' })])).sources).toBe('failed');
  });
  it('draws the JOINS from the stage: waiting on the worker, held on a person', () => {
    const worker: Blocker = { what: 'skill x is at tests', retryable: true, owner: 'worker', remedy: 'wait' };
    const operator: Blocker = { what: 'skill x: skill.tests job abc is errored', retryable: true, owner: 'operator', remedy: 're-run' };
    expect(states(at('skills', [], [worker])).skills).toBe('waiting');
    expect(states(at('skills', [], [operator])).skills).toBe('held');
    expect(states(at('rebuild', [], [])).rebuild).toBe('running');
    const cells = railCells(at('skills'), NOW);
    expect(cells.filter((c) => c.join).map((c) => c.stage)).toEqual(['skills', 'rebuild']);
  });
  it('holds clarify for the licence question and for a person, and a vocab whose job is done but is HELD', () => {
    const lic: Blocker = { what: 'licence has not been answered', field: 'licence', why: 'an unknown licence is a blocker' };
    expect(states(at('clarify', [], [lic])).clarify).toBe('held');
    expect(states(at('clarify', [], [], 'held in clarify: a restriction')).clarify).toBe('held');
    const heldVocab: Blocker = { what: 'the vocabulary is HELD: ...', retryable: false, owner: 'operator', remedy: 'force it' };
    expect(states(at('vocab', [job({ stage: 'vocab', state: 'done' })], [heldVocab])).vocab).toBe('held');
  });
  it('marks every stage done once complete', () => {
    expect(Object.values(states(at('complete'))).every((x) => x === 'done')).toBe(true);
  });
});

describe('what the card offers', () => {
  it('shows the waiting reason only while it waits', () => {
    expect(waitingLine({ waiting: 'waiting for an idle stack: the last client request was 3 min ago', stage: 'index' })).toContain('idle stack');
    expect(waitingLine({ waiting: null, stage: 'index' })).toBeNull();
    expect(waitingLine({ waiting: 'x', stage: 'complete' })).toBeNull();
  });
  it('asks for the licence only in clarify, and offers the advance only when held for a person', () => {
    const lic: Blocker = { what: 'licence has not been answered', field: 'licence' };
    expect(clarifyNeeds({ stage: 'clarify', missing: [lic], blockers: [lic], held: null }).licence).toEqual(lic);
    expect(clarifyNeeds({ stage: 'clarify', missing: [], blockers: [], held: 'held in clarify' })).toEqual({ licence: null, advance: 'held in clarify' });
    expect(clarifyNeeds({ stage: 'index', missing: [lic], blockers: [], held: 'stale' })).toEqual({ licence: null, advance: null });
  });
  it('finds the errored job a blocker names, and the errored rows to re-run (newest per stage)', () => {
    expect(erroredJobOf({ what: 'package.index job 4f2a91 is errored' })).toBe('4f2a91');
    expect(erroredJobOf({ what: 'skill koota-queries (abc): skill.tests job 77e0 is errored' })).toBe('77e0');
    expect(erroredJobOf({ what: 'package.index job 4f2a91 is queued, not done' })).toBeNull();
    const rows = [job({ id: 'a', stage: 'sources', state: 'errored', created: 1 }), job({ id: 'b', stage: 'sources', state: 'errored', created: 2 }), job({ id: 'c', stage: 'index', state: 'done' })];
    expect(rerunnableJobs(rows).map((j) => j.id)).toEqual(['b']);
  });
  it('offers FORCE only on a held vocabulary, and tier 3 only when pages wait', () => {
    expect(canForce({ state: 'held' })).toBe(true);
    expect(canForce({ state: 'promoted' })).toBe(false);
    expect(canForce(null)).toBe(false);
    expect(tier3Pending({ tier3: { pages: [{ url: 'a' }, { url: 'b' }] } })).toBe(2);
    expect(tier3Pending({ tier3: { pages: [{ url: 'a' }], ingested_by: 'operator:x' } })).toBe(0);
    expect(tier3Pending(null)).toBe(0);
  });
  it('lists the names existing packages lose, most first', () => {
    expect(lostNames({ diff: { lost: { three: ['Vector3'], koota: ['trait', 'world'], drei: [] } } })).toEqual([
      { package: 'koota', names: ['trait', 'world'] },
      { package: 'three', names: ['Vector3'] },
    ]);
  });
  it('labels a package, or the onboarding before resolve answered', () => {
    expect(packageLabel({ package: 'koota', version: '0.6.6', id: 'd1' })).toBe('koota@0.6.6');
    expect(packageLabel({ package: null, version: null, id: 'd1' })).toBe('onboarding d1');
  });
});

describe('records whose shape the server owns', () => {
  it("reads k's curve in its three shapes", () => {
    expect(curveRows([{ k: 1, accuracy: 0.5, n: 40 }, { k: 3, accuracy: 0.625, n: 40 }])).toEqual([
      { k: '1', value: '0.500', n: 40 },
      { k: '3', value: '0.625', n: 40 },
    ]);
    expect(curveRows({ '1': 0.5, '2': 0.75 })).toEqual([
      { k: '1', value: '0.500', n: null },
      { k: '2', value: '0.750', n: null },
    ]);
    expect(curveRows([0.4, 0.6]).map((c) => c.k)).toEqual(['1', '2']);
    expect(curveRows(undefined)).toEqual([]);
  });
  it('gives every evaluation number its n, from its own object or the nearest above, and lifts in_sample', () => {
    const ev = evalView({
      in_sample: true,
      in_sample_why: 'retagged after the first result',
      detection: { stripped: { n: 42, precision: 0.9, recall: 0.8 }, kept: { recall: 1 } },
      per_package: [{ package: 'koota', n: 17, recall: 0.7 }],
      git: 'abc',
      loose: 0.5,
    });
    expect(ev.inSample).toBe(true);
    expect(ev.inSampleWhy).toBe('retagged after the first result');
    const row = (w: string, m: string) => ev.rows.find((r) => r.where === w && r.measure === m);
    expect(row('detection.stripped', 'precision')).toMatchObject({ value: '0.900', n: 42, nFrom: 'detection.stripped' });
    expect(row('detection.kept', 'recall')).toMatchObject({ n: null });
    expect(row('per_package[koota]', 'recall')).toMatchObject({ n: 17 });
    expect(row('(top)', 'loose')).toMatchObject({ n: null, num: true });
    expect(row('(top)', 'git')).toMatchObject({ num: false });
    expect(ev.rows.some((r) => r.measure === 'n')).toBe(false);
    expect(evalView(null)).toEqual({ inSample: null, inSampleWhy: null, rows: [] });
    expect(evalView({ in_sample: false, x: { n: 3, y: 1 } }).inSample).toBe(false);
  });
  it('lists a record\'s scalar fields only', () => {
    expect(scalars({ ok: true, bytes: 12, nested: { a: 1 }, list: [1], none: null, skip: 'x' }, ['skip'])).toEqual([
      ['ok', 'yes'],
      ['bytes', '12'],
      ['none', '—'],
    ]);
  });
});

describe('the onboarding route', () => {
  const parser = (p: string) => {
    const keys: string[] = [];
    const src = p.replace(/:(\w+)/g, (_, k: string) => {
      keys.push(k);
      return '([^/]+)';
    });
    return { pattern: new RegExp(`^${src}/?$`, 'i'), keys };
  };
  it('reads /skills/onboarding/<id>, and keeps /skills/<id> a skill', () => {
    expect(resolve(parser, '/skills/onboarding/ds_3f9a')).toEqual({ name: 'onboarding', id: 'ds_3f9a' });
    expect(resolve(parser, '/skills/6b07a56b8c84')).toEqual({ name: 'skill', id: '6b07a56b8c84' });
    expect(href.onboarding('ds 1')).toBe('/skills/onboarding/ds%201');
  });
});

describe('renders', () => {
  it('draws the rail with the JOINS marked and a held stage as needing you', () => {
    const lic: Blocker = { what: 'licence has not been answered', field: 'licence' };
    const html = renderToStaticMarkup(createElement(OnboardingRail, { d: at('clarify', [job({})], [lic]), now: NOW }));
    expect(html).toContain('aria-label="onboarding stages"');
    expect(html).toContain('needs you');
    expect((html.match(/>join</g) ?? []).length).toBe(2);
  });
  it('draws the whole onboarding page from a full record, one panel per stage', async () => {
    const d: OnboardingDetail = {
      id: 'ds_koota',
      group: null,
      package: 'koota',
      version: '0.6.6',
      stage: 'vocab',
      state: 'waiting',
      updated: NOW - 60,
      created: NOW - 600,
      counts: {},
      jobs: { done: 3, total: 4, errored: 0, queued: 0, running: 0 },
      prompt: 'Add koota 0.6.6 https://github.com/pmndrs/koota',
      name: 'koota@0.6.6',
      licence_value: 'MIT',
      stages: STAGES,
      joins: JOINS,
      job_rows: [job({}), job({ id: 'j2', queue: 'package.index', stage: 'index', lane: 'gpu', created: 2 }), job({ id: 'j3', queue: 'package.vocab', stage: 'vocab', lane: 'cpu', created: 3, result: { state: 'held' } })],
      blockers: [{ what: 'the vocabulary is HELD: loses three on r12', retryable: false, owner: 'operator', remedy: 'force it with POST /dash/api/skill-factory/onboarding/promote {id}' }],
      missing: [],
      warnings: [],
      next_stage: 'examples',
      field_states: [],
      held: null,
      waiting: null,
      notes: { links: [], aliases: ['koota'], replaces: null, author: 'operator:abc' },
      resolution: {
        links: ['https://github.com/pmndrs/koota'],
        package: { ecosystem: 'npm', name: 'koota', version: '0.6.6', version_rule: 'prompt', version_why: 'the prompt says 0.6.6', commit: 'a'.repeat(40), commit_rule: 'gitHead', repository: { owner: 'pmndrs', repo: 'koota' }, links: ['https://github.com/pmndrs/koota'], licence_quotes: [] },
        sources: [],
        unresolved: [],
        how: "resolved from the prompt's links",
        replaces: { mode: 'new', old: [], why: 'no version of koota is held' },
        siblings: [],
      },
      licence: { quotes: [{ spdx: 'MIT', quote: 'MIT License', where: 'LICENSE at aaaa', kind: 'file' }], chosen: { spdx: 'MIT', quote: 'MIT License', where: 'LICENSE' } },
      index: { fetch: { ok: true, tarball_bytes: 123 }, health: { ok: true, chunks: 88 } },
      vocab: { state: 'held', why: 'the standing detection labels get worse', diff: { per_package: { koota: { names: 40, unique: 31, code_only: 2 } }, lost: { three: ['world'] }, gained: {} }, floor: { passed: false, rows: 120, lost_tp: [{ id: 'r12', package: 'three' }], gained_fp: [] } },
      examples: { example_groups: 12, example_files: 30, per_package: { koota: { train: 9, held_out: 3 } } },
      knn: { k: 3, k_n: 9, groups: 9, curve: [{ k: 1, accuracy: 0.5, n: 9 }, { k: 3, accuracy: 0.66, n: 9 }] },
      sources: { chosen: [{ tier: 1, path: 'skills/koota/SKILL.md', frontier: true, why: 'the package ships a SKILL.md' }], skipped: [], tier3: { where: 'https://koota.dev/llms.txt', pages: [{ url: 'https://koota.dev/a' }] }, made: { created: ['s1'], failed: [] } },
      skills: [{ id: 's1', name: 'koota-traits-and-entities', status: 'armed', version: 1, latest_version: 1, reason: null, lead_for: 'koota', parent: null, source_url: null }],
      skill_counts: { armed: 1 },
      retire: null,
      eval: { in_sample: false, detection: { stripped: { n: 30, recall: 0.8 } } },
      reviews: [{ stage: 'vocab', note: 'the loss of world is real', by: 'operator:abc', at: NOW - 30 }],
    };
    const { fetchShared } = await import('../api/cache');
    const { onboardingPath } = await import('../api/onboarding');
    const { OnboardingScreen } = await import('./Onboarding');
    const { Router } = await import('wouter');
    const g = globalThis as unknown as { fetch: unknown; localStorage: unknown };
    const saved = { fetch: g.fetch, localStorage: g.localStorage };
    g.localStorage = { getItem: () => 'test-key', setItem: () => undefined, removeItem: () => undefined };
    g.fetch = async () => new Response(JSON.stringify({ onboarding: d }), { status: 200 });
    try {
      const r = await fetchShared(onboardingPath(d.id));
      expect(r.ok).toBe(true);
    } finally {
      g.fetch = saved.fetch;
      g.localStorage = saved.localStorage;
    }
    const html = renderToStaticMarkup(createElement(Router, { ssrPath: href.onboarding(d.id), children: createElement(OnboardingScreen, { id: d.id }) }));
    for (const want of ['koota@0.6.6', 'RESOLVE', 'CLARIFY', 'VOCAB', 'EVALUATE', 'REBUILD', 'gitHead', 'MIT License', 'FORCE THE VOCABULARY', 'world', 'INGEST 1 TIER-3 PAGE(S)', 'koota-traits-and-entities', 'HELD OUT (not in-sample)', 'the loss of world is real', 'NEEDS YOU']) {
      expect(html, want).toContain(want);
    }
    expect(html).toContain('href="/skills/s1"');
  });
  it('prints each blocker with its remedy and owner, and a re-run only where a job errored', () => {
    const html = renderToStaticMarkup(
      createElement(Blockers, {
        blockers: [
          { what: 'package.index job 4f2a91 is errored', why: 'embeddings 503', retryable: true, owner: 'operator', remedy: 'fix the cause, then re-run that job' },
          { what: 'the skill document index is not fresh (stale)', retryable: true, owner: 'worker', remedy: 'its upkeep job rebuilds it when the stack is idle' },
        ],
        busy: false,
        onRerun: () => undefined,
      }),
    );
    expect(html).toContain('REMEDY (operator) fix the cause');
    expect(html).toContain('REMEDY (worker) its upkeep job');
    expect(html).toContain('embeddings 503');
    expect((html.match(/RE-RUN JOB/g) ?? []).length).toBe(1);
  });
});
