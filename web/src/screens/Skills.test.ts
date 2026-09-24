// The skills screen's pure helpers, and a static render of the list (vitest
// runs in node: react-dom/server).
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { appliesToText, createBody, overallFallbackRate, statusTone, versionText, type SkillSummary } from '../api/skills';
import { SkillList } from './Skills';

const skill = (over: Partial<SkillSummary>): SkillSummary => ({
  id: 'abc123def456',
  name: 'ts',
  title: 'TypeScript Best Practices',
  source_url: null,
  source_kind: 'text',
  status: 'armed',
  reason: null,
  enabled: true,
  served_version: 2,
  latest_version: 3,
  watch_seconds: null,
  next_watch: null,
  created: 0,
  updated: 0,
  text: 'TypeScript Best Practices\napplies when: TypeScript is being used\n- DO: x',
  text_version: 2,
  applies_when: 'TypeScript is being used',
  applies_to: { artifacts: ['code'], languages: ['typescript'], frameworks: [] },
  triggers: [],
  meta: {},
  ...over,
});

describe('skills helpers', () => {
  it('tones a status', () => {
    expect(statusTone('armed')).toBe('moss');
    expect(statusTone('quarantined')).toBe('crimson');
    expect(statusTone('failed')).toBe('crimson');
    expect(statusTone('disabled')).toBe('muted');
    expect(statusTone('pipeline')).toBe('cyan');
  });
  it('says which version serves', () => {
    expect(versionText({ served_version: 2, latest_version: 3 })).toBe('v2 of 3');
    expect(versionText({ served_version: null, latest_version: 1 })).toBe('none of 1');
  });
  it('sends a URL as a url and anything else as text', () => {
    expect(createBody(' https://example.com/SKILL.md ')).toEqual({ url: 'https://example.com/SKILL.md' });
    expect(createBody('# A skill\n- DO: x')).toEqual({ text: '# A skill\n- DO: x' });
    expect(createBody('see https://example.com and more')).toEqual({ text: 'see https://example.com and more' });
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

describe('SkillList', () => {
  it('renders each skill with its status and served version', () => {
    const html = renderToStaticMarkup(createElement(SkillList, { skills: [skill({}), skill({ id: 'q', status: 'quarantined', title: 'Bad Skill', served_version: null })], onPick: () => {} }));
    expect(html).toContain('TypeScript Best Practices');
    expect(html).toContain('v2 of 3');
    expect(html).toContain('quarantined');
    expect(html).toContain('none of 3');
  });
  it('says so when there are none', () => {
    const html = renderToStaticMarkup(createElement(SkillList, { skills: [], onPick: () => {} }));
    expect(html).toContain('no skills');
  });
});
