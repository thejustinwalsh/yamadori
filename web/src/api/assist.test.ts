import { describe, expect, it } from 'vitest';
import { answerBody, badgeFor, BADGES, displayValue, draftOf, linkable, pollEvery, refusalText, submissionBody, waitingOn } from './assist';
import type { Dataset, FieldState, JobRow } from './types';

const field = (over: Partial<FieldState>): FieldState => ({
  field: 'licence',
  label: 'LICENCE',
  input: 'text',
  severity: 'blocker',
  why: 'because',
  value: '',
  assisted: true,
  state: 'needs_you',
  ...over,
});

const job = (state: string, over: Partial<JobRow> = {}): JobRow => ({
  id: 'j1',
  queue: 'dataset.assist',
  lane: 'gpu',
  state,
  stage: 'clarify',
  attempts: 1,
  max_attempts: 3,
  error: null,
  progress: null,
  created: 0,
  ...over,
});

describe('submissionBody', () => {
  it('sends one http(s) URL as {url}', () => {
    expect(submissionBody('  https://example.com/docs  ', 'recipes')).toEqual({ url: 'https://example.com/docs', kind: 'recipes' });
  });
  it('sends anything else as {text}, including a URL with prose around it', () => {
    expect(submissionBody('see https://example.com/docs', 'laya')).toEqual({ text: 'see https://example.com/docs', kind: 'laya' });
    expect(submissionBody('line one\nline two', 'recipes')).toEqual({ text: 'line one\nline two', kind: 'recipes' });
  });
  it('refuses an empty paste with a reason rather than posting it', () => {
    expect(submissionBody('   ', 'recipes')).toHaveProperty('error');
  });
});

describe('badges', () => {
  it('has one distinct word per provenance', () => {
    expect(badgeFor('evidence').label).toBe('EVIDENCE');
    expect(badgeFor('proposed').label).toBe('PROPOSED');
    expect(badgeFor('needs_you').label).toBe('NEEDS YOU');
    expect(badgeFor('operator').label).toBe('OPERATOR');
    const labels = Object.values(BADGES).map((b) => b.label);
    expect(new Set(labels).size).toBe(labels.length);
  });
  it('treats an unknown state as NEEDS YOU, never as evidence', () => {
    expect(badgeFor('something-new').label).toBe('NEEDS YOU');
  });
  it('never uses the evidence colour for a proposal', () => {
    expect(BADGES.proposed.tone).not.toBe(BADGES.evidence.tone);
  });
});

describe('values and answers', () => {
  it('says "not answered" in words for every empty shape', () => {
    expect(displayValue(null)).toBe('not answered');
    expect(displayValue('')).toBe('not answered');
    expect(displayValue([])).toBe('not answered');
    expect(displayValue(['gpu', 'types'])).toBe('gpu, types');
  });
  it('answers a multi field with a list and a text field with a string', () => {
    const d = field({ field: 'domains', input: 'multi', value: [] });
    expect(answerBody('abc', d, ['gpu'])).toEqual({ id: 'abc', domains: ['gpu'] });
    expect(answerBody('abc', field({}), '  CC-BY-4.0 ')).toEqual({ id: 'abc', licence: 'CC-BY-4.0' });
  });
  it('starts an edit from the current value, not a suggestion', () => {
    expect(draftOf(field({ value: 'MIT' }))).toBe('MIT');
    expect(draftOf(field({ field: 'domains', input: 'multi', value: ['gpu'] }))).toEqual(['gpu']);
  });
  it('links only http(s) locations', () => {
    expect(linkable('https://x.test/LICENSE')).toBe(true);
    expect(linkable('the pasted source text')).toBe(false);
    expect(linkable('javascript:alert(1)')).toBe(false);
  });
});

describe('polling and status', () => {
  it('polls fast only while a job is queued or running', () => {
    expect(pollEvery({ job_rows: [job('running')] })).toBe(3000);
    expect(pollEvery({ job_rows: [job('done'), job('errored')] })).toBe(15000);
  });
  it('says what the dataset is waiting on, from the payload only', () => {
    const base = { stage: 'clarify', job_rows: [job('done')] } as unknown as Dataset;
    expect(waitingOn({ ...base, job_rows: [job('running', { progress: 'asking the model' })] })).toBe('assist running: asking the model');
    expect(waitingOn({ ...base, field_states: [field({}), field({ field: 'language' })] })).toBe('2 fields need you');
    expect(waitingOn({ ...base, field_states: [field({ state: 'evidence' })], assist: { held: 'held in clarify: AGPL' } })).toBe('held: a person decides');
    expect(waitingOn({ ...base, job_rows: [job('errored')] })).toBe('a job errored');
  });
  it('prints a refusal with every reason the server gave', () => {
    expect(
      refusalText({ failure: { kind: 'http', status: 409, message: 'cannot leave clarify' }, reasons: [{ what: 'licence has not been answered' }] }),
    ).toBe('HTTP 409: cannot leave clarify -- licence has not been answered');
  });
});
