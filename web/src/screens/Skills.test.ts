// The skill factory's pure helpers, the router's /skills paths, and static
// renders of the library table and the SKILL.md renderer (vitest runs in
// node: react-dom/server).
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { Router } from 'wouter';
import {
  appliesToText,
  canRerun,
  facetCounts,
  facetsOf,
  filterSkills,
  groupDecisions,
  isSkillMd,
  overallFallbackRate,
  overCaps,
  sizeTone,
  splitRemedy,
  splitSkillMd,
  stageStates,
  statusTone,
  submitBody,
  toggle,
  versionText,
  type Limits,
  type SkillSummary,
} from '../api/skills';
import { href, resolve } from '../router';
import { Markdown } from './skills/parts';
import { SkillTable } from './Skills';

const LIMITS: Limits = {
  label: 'choices',
  chars_per_token: 3,
  skill_tokens_aim: [100, 300],
  skill_tokens_hard: 450,
  max_items: 6,
  max_item_chars: 300,
  max_prohibitions: 2,
  name_chars: 64,
  description_chars: 1024,
  max_skills_per_turn: 3,
};

const TAXONOMY = {
  artifact: [{ id: 'code', name: 'code' }],
  language: [
    { id: 'typescript', name: 'TypeScript' },
    { id: 'wgsl', name: 'WGSL' },
  ],
  framework: [{ id: 'react', name: 'React' }],
  phase: ['plan', 'implement', 'debug'],
  situation: ['error_output'],
  tools: "the client's tool names",
};

const skill = (over: Partial<SkillSummary>): SkillSummary => ({
  id: 'abc123def456',
  name: 'ts-best-practices',
  title: 'TypeScript Best Practices',
  source_url: null,
  source_kind: 'migration',
  status: 'armed',
  reason: null,
  enabled: true,
  served_version: 2,
  latest_version: 3,
  watch_seconds: null,
  next_watch: null,
  created: 0,
  updated: 0,
  meta: {},
  text_version: 2,
  description: 'Use when writing TypeScript.',
  tags: ['TypeScript'],
  provenance: { kind: 'migration', licence: { spdx: 'MIT', quote: 'MIT License' } },
  applies_when: 'TypeScript is being used',
  applies_to: { artifacts: ['code'], languages: ['typescript'], frameworks: [] },
  category: { artifact: ['code'], language: ['typescript'], framework: [], phase: [], domain: ['types'] },
  gates: { phases: [], tools_any: [], tools_all: [], tools_none: [], situations: [], all_of: [], topics: ['satisfies'] },
  triggers: [],
  activation: { passed: true, score: 1, n: 4, failures: [] },
  folder: 'ts-best-practices',
  size: { tokens: 207, chars: 619, items: 4, prohibitions: 1, longest_item_chars: 189, description_chars: 173 },
  ...over,
});

describe('skills helpers', () => {
  it('tones a status', () => {
    expect(statusTone('armed')).toBe('moss');
    expect(statusTone('quarantined')).toBe('crimson');
    expect(statusTone('failed')).toBe('crimson');
    expect(statusTone('disabled')).toBe('muted');
    expect(statusTone('archived')).toBe('muted');
    expect(statusTone('pipeline')).toBe('cyan');
  });
  it('says which version serves', () => {
    expect(versionText({ served_version: 2, latest_version: 3 })).toBe('v2 of 3');
    expect(versionText({ served_version: null, latest_version: 1 })).toBe('none of 1');
  });
  it('reports the fallback rate over days, or null with no candidates', () => {
    const day = (c: number, f: number) => ({ day: 'd', requests: c, with_candidates: c, injected: 0, fallbacks: f, fallback_errors: 0, cache_hits: 0, fallback_rate: null });
    expect(overallFallbackRate([day(4, 1), day(6, 2)])).toBeCloseTo(0.3);
    expect(overallFallbackRate([day(0, 0)])).toBeNull();
  });
  it('lists what a skill applies to', () => {
    expect(appliesToText({ artifacts: ['tests'], languages: ['typescript'], frameworks: ['vitest'] })).toBe('tests, vitest, typescript');
    expect(appliesToText(null)).toBe('—');
  });
});

describe('the library facets', () => {
  const facets = facetsOf({ taxonomy: TAXONOMY });
  const ts = skill({});
  const wgsl = skill({ id: 'w', title: 'Texture Bindings', description: 'Use when a shader binds a texture.', applies_when: 'WGSL is being used', category: { artifact: ['code'], language: ['wgsl'], framework: [], phase: ['debug'], domain: ['gpu'] }, provenance: { licence: { spdx: 'BSD-3-Clause' } }, tags: ['WGSL'], gates: { phases: [], tools_any: [], tools_all: ['terminal'], tools_none: ['view_image'], situations: [], all_of: [], topics: ['rgba32float'] } });
  const off = skill({ id: 'q', status: 'quarantined', title: 'Bad', provenance: null });
  const all = [ts, wgsl, off];
  it('names values from the server taxonomy', () => {
    expect(facets.find((f) => f.key === 'language')!.names.typescript).toBe('TypeScript');
    expect(facets.find((f) => f.key === 'phase')!.names.debug).toBe('debug');
  });
  it('ORs within a facet and ANDs across facets', () => {
    expect(filterSkills(all, facets, { language: ['typescript', 'wgsl'] }, '').map((s) => s.id)).toEqual(['abc123def456', 'w', 'q']);
    expect(filterSkills(all, facets, { language: ['typescript'], status: ['armed'] }, '').map((s) => s.id)).toEqual(['abc123def456']);
    expect(filterSkills(all, facets, { tools: ['none:view_image'] }, '').map((s) => s.id)).toEqual(['w']);
    expect(filterSkills(all, facets, { licence: ['none recorded'] }, '').map((s) => s.id)).toEqual(['q']);
  });
  it('searches titles, tags and topics, every word', () => {
    expect(filterSkills(all, facets, {}, 'rgba32float').map((s) => s.id)).toEqual(['w']);
    expect(filterSkills(all, facets, {}, 'texture wgsl').map((s) => s.id)).toEqual(['w']);
    expect(filterSkills(all, facets, {}, 'texture typescript')).toEqual([]);
  });
  it("counts a facet's values against the other facets only", () => {
    const f = facets.find((x) => x.key === 'language')!;
    const counts = Object.fromEntries(facetCounts(all, facets, { language: ['wgsl'], status: ['armed'] }, '', f));
    expect(counts).toEqual({ typescript: 1, wgsl: 1 });
  });
  it('toggles a value on and off', () => {
    const on = toggle({}, 'phase', 'debug');
    expect(on).toEqual({ phase: ['debug'] });
    expect(toggle(on, 'phase', 'debug')).toEqual({});
  });
});

describe('sizes against the caps', () => {
  it('tones by aim and hard cap', () => {
    expect(sizeTone(207, LIMITS)).toBe('moss');
    expect(sizeTone(301, LIMITS)).toBe('cyan');
    expect(sizeTone(451, LIMITS)).toBe('crimson');
  });
  it('names every cap a size breaks', () => {
    expect(overCaps({ tokens: 200, chars: 600, items: 4, prohibitions: 1, longest_item_chars: 100, description_chars: 100 }, LIMITS)).toEqual([]);
    const over = overCaps({ tokens: 500, chars: 1500, items: 7, prohibitions: 3, longest_item_chars: 301, description_chars: 1025 }, LIMITS);
    expect(over).toHaveLength(5);
    expect(over[2]).toBe('3 prohibitions, over 2');
  });
});

describe('the pipeline', () => {
  const stages = ['fetch', 'screen', 'screen_model', 'licence', 'distil', 'decompose', 'classify', 'tests', 'validate', 'arm'];
  const path = ['screen', 'screen_model', 'licence', 'distil', 'classify', 'tests', 'validate', 'arm'];
  it('walks a version over its path', () => {
    const st = stageStates(stages, { path, stage: 'licence', state: 'failed' });
    expect(st.map((x) => x.state)).toEqual(['skipped', 'done', 'done', 'failed', 'pending', 'skipped', 'pending', 'pending', 'pending', 'pending']);
    const run = stageStates(stages, { path, stage: 'distil', state: 'running' }, [{ id: 'j', queue: 'skill.distil', state: 'queued', stage: 'distil', error: null, progress: null }]);
    expect(run.find((x) => x.stage === 'distil')!.state).toBe('queued');
    expect(stageStates(stages, { path, stage: 'arm', state: 'armed' }).at(-1)!.state).toBe('armed');
    expect(stageStates(stages, { path, stage: 'screen', state: 'planned' }).filter((x) => x.state === 'pending')).toHaveLength(path.length);
  });
  it('splits a failure into its situation and remedy', () => {
    const a = splitRemedy('licence: no licence could be established (searched x). Remedy (operator): POST /dash/api/skill/licence {id, licence, quote}, then re-run the licence stage');
    expect(a.situation).toBe('licence: no licence could be established (searched x)');
    expect(a.remedy).toBe('(operator) POST /dash/api/skill/licence {id, licence, quote}, then re-run the licence stage');
    const b = splitRemedy('fetch timed out | retryable: yes | remedy (worker): it retries');
    expect(b).toEqual({ situation: 'fetch timed out', retryable: 'yes', remedy: '(worker) it retries' });
    expect(splitRemedy('screen: hidden_html')).toEqual({ situation: 'screen: hidden_html', retryable: null, remedy: null });
  });
  it('re-runs only a version that stopped', () => {
    expect(canRerun({ state: 'failed' })).toBe(true);
    expect(canRerun({ state: 'armed' })).toBe(false);
    expect(canRerun({ state: 'decomposed' })).toBe(false);
    expect(canRerun(undefined)).toBe(false);
  });
});

describe('submitting', () => {
  const f = { mode: 'text' as const, input: '', goal: '', name: '', watchHours: '' };
  it('sends pasted text with its goal', () => {
    expect(submitBody({ ...f, input: 'Prefer small pure functions.', goal: ' a skill for TS style ' })).toEqual({ body: { text: 'Prefer small pure functions.', goal: 'a skill for TS style' } });
  });
  it('sends one URL as url and several as urls', () => {
    expect(submitBody({ ...f, mode: 'urls', input: ' https://x.dev/a/SKILL.md ', watchHours: '0' })).toEqual({ body: { url: 'https://x.dev/a/SKILL.md', watch_hours: 0 } });
    expect(submitBody({ ...f, mode: 'urls', input: 'https://a.dev/x\nhttps://b.dev/y', name: 'n' })).toEqual({ body: { urls: ['https://a.dev/x', 'https://b.dev/y'] } });
    expect(submitBody({ ...f, mode: 'urls', input: 'https://a.dev not-a-url' })).toHaveProperty('error');
    expect(submitBody({ ...f, mode: 'urls', input: Array.from({ length: 26 }, (_, i) => `https://a.dev/${i}`).join('\n') })).toHaveProperty('error');
  });
  it('forces the frontier path only for a SKILL.md', () => {
    const md = '---\nname: pdf\ndescription: Use when reading PDFs\n---\n# PDF\n';
    expect(isSkillMd(md)).toBe(true);
    expect(isSkillMd('# just a doc')).toBe(false);
    expect(submitBody({ ...f, mode: 'frontier', input: md })).toEqual({ body: { text: md, frontier: true } });
    expect(submitBody({ ...f, mode: 'frontier', input: '# no frontmatter' })).toHaveProperty('error');
  });
});

describe('selections', () => {
  it('groups the rows of one decision', () => {
    const row = (ts: number, skill: string, request: string) => ({ ts, skill, version: 1, route_class: 'code_edit', decided_by: 'deterministic', strength: 'fact', request });
    const d = groupDecisions([row(1, 'a', 'r1'), row(2, 'b', 'r2'), row(1, 'c', 'r1')]);
    expect(d.map((x) => [x.ts, x.rows.map((r) => r.skill)])).toEqual([
      [2, ['b']],
      [1, ['a', 'c']],
    ]);
  });
});

describe('routes', () => {
  const parser = (p: string) => {
    const keys: string[] = [];
    const src = p.replace(/:(\w+)/g, (_, k: string) => {
      keys.push(k);
      return '([^/]+)';
    });
    return { pattern: new RegExp(`^${src}/?$`, 'i'), keys };
  };
  it('reads the factory views and a skill page', () => {
    expect(resolve(parser, '/skills')).toEqual({ name: 'skills', view: 'library' });
    expect(resolve(parser, '/skills/create')).toEqual({ name: 'skills', view: 'create' });
    expect(resolve(parser, '/skills/selections')).toEqual({ name: 'skills', view: 'selections' });
    expect(resolve(parser, '/skills/6b07a56b8c84')).toEqual({ name: 'skill', id: '6b07a56b8c84' });
    expect(href.skill('abc')).toBe('/skills/abc');
    expect(href.skillsView('library')).toBe('/skills');
  });
});

describe('renders', () => {
  it('lists skills with state, size and prohibition count', () => {
    const html = renderToStaticMarkup(
      createElement(Router, {
        ssrPath: '/skills',
        children: createElement(SkillTable, { skills: [skill({}), skill({ id: 'q', status: 'quarantined', title: 'Bad Skill', size: { tokens: 500, chars: 1500, items: 7, prohibitions: 3, longest_item_chars: 1, description_chars: 1 } })], o: { limits: LIMITS, taxonomy: TAXONOMY } }),
      }),
    );
    expect(html).toContain('TypeScript Best Practices');
    expect(html).toContain('href="/skills/abc123def456"');
    expect(html).toContain('quarantined');
    expect(html).toContain('207/450 TOK');
    expect(html).toContain('4/6');
    expect(html).toContain('3/2');
  });
  it('says so when nothing matches', () => {
    const html = renderToStaticMarkup(createElement(SkillTable, { skills: [], o: { limits: LIMITS, taxonomy: TAXONOMY } }));
    expect(html).toContain('no skill matches');
  });
  it('renders a SKILL.md body without parsing it as HTML', () => {
    const { frontmatter, body } = splitSkillMd('---\nname: x\n---\n# Title\n\n## Guidance\n\n- DO: call `useActionState`\n- DO NOT: <script>alert(1)</script>\n');
    expect(frontmatter).toBe('name: x');
    const html = renderToStaticMarkup(createElement(Markdown, { md: body }));
    expect(html).toContain('<code');
    expect(html).toContain('useActionState');
    expect(html).toContain('DO NOT:');
    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;');
  });
});
