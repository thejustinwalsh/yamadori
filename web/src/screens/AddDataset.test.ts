// Static renders of the ADD DATASET form and the field badges. The vitest
// environment is node, so these render to markup with react-dom/server; the
// interactive half (what a submit or a save POSTs) is the pure functions in
// src/api/assist.test.ts.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { FieldState } from '../api/types';
import { AddDatasetPanel, FieldRow } from './AddDataset';

const field = (over: Partial<FieldState>): FieldState => ({
  field: 'licence',
  label: 'LICENCE',
  input: 'text',
  severity: 'blocker',
  why: 'This repo has already had to flag an AGPL corpus.',
  value: '',
  assisted: true,
  state: 'needs_you',
  ...over,
});

const row = (f: FieldState, canEdit = true) =>
  renderToStaticMarkup(createElement(FieldRow, { datasetId: 'abc', f, onSaved: () => {}, canEdit }));

describe('AddDatasetPanel', () => {
  const html = renderToStaticMarkup(createElement(AddDatasetPanel, { onSubmitted: () => {} }));
  it('asks for a URL or pasted text, and a kind', () => {
    expect(html).toContain('URL OR PASTED TEXT');
    expect(html).toContain('<textarea');
    expect(html).toContain('value="recipes"');
    expect(html).toContain('value="laya"');
  });
  it('cannot submit an empty paste', () => {
    expect(html).toMatch(/<button type="submit" disabled=""/);
  });
  it('says the licence is quoted or asked, never guessed', () => {
    expect(html).toContain('never guessed');
  });
});

describe('FieldRow badges', () => {
  it('EVIDENCE shows the quote and links where it was found', () => {
    const html = row(
      field({ state: 'evidence', value: 'CC-BY-4.0', quote: 'This documentation is licensed under CC-BY-4.0.', found_in: 'https://x.test/docs' }),
    );
    expect(html).toContain('EVIDENCE');
    expect(html).toContain('This documentation is licensed under CC-BY-4.0.');
    expect(html).toContain('href="https://x.test/docs"');
    expect(html).toContain('data-state="evidence"');
    expect(html).not.toContain('NEEDS YOU');
  });
  it('EVIDENCE from pasted text names it without inventing a link', () => {
    const html = row(field({ state: 'evidence', value: 'MIT License', quote: 'MIT License', found_in: 'the pasted source text' }));
    expect(html).toContain('the pasted source text');
    expect(html).not.toContain('href="the pasted');
  });
  it('PROPOSED says it is the model and not a quote', () => {
    const html = row(field({ field: 'language', label: 'LANGUAGE OR AREA', state: 'proposed', value: 'rust', basis: "the model's reading of the source" }));
    expect(html).toContain('PROPOSED');
    expect(html).toContain('Not a quote');
    expect(html).not.toContain('<blockquote');
  });
  it('NEEDS YOU carries the why and what was already searched', () => {
    const html = row(field({ searched: 'the assist searched https://x.test/page; https://x.test/LICENSE (HTTP 404)' }));
    expect(html).toContain('NEEDS YOU');
    expect(html).toContain('BLOCKER');
    expect(html).toContain('This repo has already had to flag an AGPL corpus.');
    expect(html).toContain('ALREADY SEARCHED');
    expect(html).toContain('HTTP 404');
    expect(html).toContain('[ ANSWER ]');
  });
  it('an answered field offers an override, and a past-clarify one offers nothing', () => {
    expect(row(field({ state: 'proposed', value: 'rust' }))).toContain('[ OVERRIDE ]');
    const ro = row(field({ state: 'operator', value: 'MIT' }), false);
    expect(ro).not.toContain('[ OVERRIDE ]');
    expect(ro).toContain('OPERATOR');
  });
  it('an operator override says what model value it replaced', () => {
    const html = row(field({ state: 'operator', value: 'CC0-1.0', overrode: { provenance: 'evidence', value: 'CC-BY-4.0' } }));
    expect(html).toMatch(/REPLACED THE MODEL(&#x27;|')S EVIDENCE VALUE: CC-BY-4.0/);
  });
});
