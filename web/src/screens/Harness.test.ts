// HARNESS TOOLS: the pure helpers (filter, grouping, the form's fields) and
// static renders of an entry card and the form. vitest runs in node, so the
// components render to markup with react-dom/server.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  ALL,
  byKind,
  emptyForm,
  fieldsOf,
  formOf,
  formProblems,
  kitFileName,
  linkOf,
  visible,
  type Entry,
  type Filter,
  type KitOverview,
} from '../api/harness';
import { resolve } from '../router';
import { EntryCard, EntryForm } from './Harness';

const base: Entry = {
  id: 'hk-1',
  seed_id: 'pi.env',
  version: 1,
  harness: 'pi',
  kind: 'config',
  title: 'Pi: environment',
  body: 'PI_CACHE_RETENTION=long',
  evidence: [{ ref: 'docs/HARNESS-PI.md s2', showed: 'prompt_cache_key = the session id (captured)' }],
  status: 'recommended',
  status_why: '',
  proof: 'verified',
  where: 'both',
  needs_operator: false,
  target: 'the environment Pi runs in',
  file_name: 'pi.env',
  skill_path: '',
  download: null,
  created: 1790700000,
  created_by: 'seed',
  updated: 1790700000,
  updated_by: 'seed',
  deleted: false,
  deleted_at: null,
  deleted_by: null,
  deleted_why: null,
};
const e = (over: Partial<Entry>): Entry => ({ ...base, body: 'other text', ...over });

const entries: Entry[] = [
  base,
  e({ id: 'hk-2', harness: 'any', kind: 'download', title: 'typescript 5.9.3', where: 'both', download: {
    source_url: 'https://www.npmjs.com/package/typescript/v/5.9.3', version: '5.9.3', hash: 'sha512-abc==', commit: '', licence: 'Apache-2.0',
    pin_missing_why: '', verify: 'npm view typescript@5.9.3 dist.integrity', install: { windows: 'npm install -g --ignore-scripts typescript@5.9.3', unix: 'npm install -g --ignore-scripts typescript@5.9.3' } } }),
  e({ id: 'hk-3', harness: 'hermes', kind: 'mcp_server', title: 'DECISION: lsmcp', status: 'trying', status_why: 'awaiting the operator', needs_operator: true, proof: 'unmeasured' }),
  e({ id: 'hk-4', harness: 'any', kind: 'note', title: 'slow WebGL', where: 'box', proof: 'unmeasured' }),
  e({ id: 'hk-5', harness: 'any', kind: 'mcp_server', title: 'cclsp', status: 'rejected', status_why: 'reported clean code when broken' }),
];
const f = (over: Partial<Filter>): Filter => ({ harness: ALL, status: '', where: '', needsOperator: false, q: '', ...over });
const kinds = [
  { id: 'prompt', label: 'prompt' },
  { id: 'skill', label: 'skill folder' },
  { id: 'download', label: 'download' },
  { id: 'mcp_server', label: 'MCP server' },
  { id: 'config', label: 'config snippet' },
  { id: 'note', label: 'note' },
];
const vocab: Pick<KitOverview, 'harnesses' | 'kinds' | 'statuses' | 'proofs' | 'wheres' | 'skill_roots' | 'tokens' | 'key_placeholder'> = {
  harnesses: [
    { id: 'hermes', label: 'Hermes' },
    { id: 'pi', label: 'Pi' },
    { id: 'any', label: 'any harness' },
  ],
  kinds,
  statuses: ['recommended', 'trying', 'rejected'],
  proofs: [
    { id: 'verified', label: 'verified' },
    { id: 'unmeasured', label: 'unmeasured' },
  ],
  wheres: [
    { id: 'remote', label: 'remote' },
    { id: 'box', label: 'box' },
    { id: 'both', label: 'both' },
  ],
  skill_roots: ['bench/sandbox/harness_skills'],
  tokens: ['{{API_BASE}}'],
  key_placeholder: '<PASTE-YOUR-YAMADORI-KEY-HERE>',
};

describe('visible', () => {
  it('a harness shows its own entries and the general ones', () => {
    expect(visible(entries, f({ harness: 'pi' })).map((x) => x.id)).toEqual(['hk-1', 'hk-2', 'hk-4', 'hk-5']);
  });
  it('"any" shows only the general entries; ALL shows everything', () => {
    expect(visible(entries, f({ harness: 'any' })).map((x) => x.id)).toEqual(['hk-2', 'hk-4', 'hk-5']);
    expect(visible(entries, f({})).length).toBe(5);
  });
  it('filters by status, machine, the operator flag and text', () => {
    expect(visible(entries, f({ status: 'rejected' })).map((x) => x.id)).toEqual(['hk-5']);
    expect(visible(entries, f({ where: 'remote' })).map((x) => x.id)).not.toContain('hk-4');
    expect(visible(entries, f({ where: 'box' })).map((x) => x.id)).toContain('hk-4');
    expect(visible(entries, f({ needsOperator: true })).map((x) => x.id)).toEqual(['hk-3']);
    expect(visible(entries, f({ q: 'CACHE_RETENTION' })).map((x) => x.id)).toEqual(['hk-1']);
    expect(visible(entries, f({ q: 'session id' })).map((x) => x.id)).toContain('hk-1');
  });
});

describe('byKind', () => {
  it('groups in the vocabulary order, pending decisions first, empty kinds dropped', () => {
    const g = byKind(entries, kinds);
    expect(g.map((x) => x.kind.id)).toEqual(['download', 'mcp_server', 'config', 'note']);
    expect(g.find((x) => x.kind.id === 'mcp_server')?.entries.map((x) => x.id)).toEqual(['hk-3', 'hk-5']);
  });
});

describe('form', () => {
  it('an entry round-trips through the form into the same fields', () => {
    const d = entries[1] as Entry;
    const fields = fieldsOf(formOf(d));
    expect(fields.download).toEqual({ ...d.download, install: d.download?.install });
    expect(fields.evidence).toEqual(d.evidence);
  });
  it('an empty download block sends null, and a non-skill sends no skill folder', () => {
    const fields = fieldsOf({ ...emptyForm('pi'), title: 'x', skill_path: 'bench/x' });
    expect(fields.download).toBeNull();
    expect(fields.skill_path).toBe('');
  });
  it('names what is still needed before sending', () => {
    const blank = emptyForm(ALL);
    expect(blank.harness).toBe('any');
    const p = formProblems(blank);
    expect(p).toContain('a title');
    expect(p.some((x) => x.startsWith('why it is trying'))).toBe(true);
    const rec = { ...blank, title: 't', status: 'recommended' as const };
    expect(formProblems(rec).some((x) => x.startsWith('evidence'))).toBe(true);
    expect(formProblems({ ...rec, evidence: [{ ref: 'docs/X.md', showed: 'it' }] })).toEqual([]);
  });
});

describe('EntryCard', () => {
  const render = (x: Entry, extra = {}) => renderToStaticMarkup(createElement(EntryCard, { e: x, harnesses: vocab.harnesses, ...extra }));
  it('shows status, proof, machine, harness, the evidence and the version', () => {
    const html = render(base);
    expect(html).toContain('RECOMMENDED');
    expect(html).toContain('VERIFIED · NO MODEL');
    expect(html).toContain('REMOTE + BOX');
    expect(html).toContain('PI');
    expect(html).toContain('docs/HARNESS-PI.md s2');
    expect(html).toContain('prompt_cache_key = the session id (captured)');
    expect(html).toContain('v1 · updated');
  });
  it('a download shows its source link, exact version, pin, licence and install per OS', () => {
    const html = render(entries[1] as Entry);
    expect(html).toContain('href="https://www.npmjs.com/package/typescript/v/5.9.3"');
    expect(html).toContain('5.9.3');
    expect(html).toContain('sha512-abc==');
    expect(html).toContain('Apache-2.0');
    expect(html).toContain('INSTALL · WINDOWS');
    expect(html).toContain('INSTALL · LINUX/MAC');
  });
  it('a pending decision is flagged, and unmeasured says so', () => {
    const html = render(entries[2] as Entry);
    expect(html).toContain('NEEDS OPERATOR');
    expect(html).toContain('UNMEASURED');
    expect(html).toContain('awaiting the operator');
  });
  it('a URL evidence ref is a link; a doc path is text', () => {
    expect(linkOf('https://github.com/oraios/serena')).toBe('https://github.com/oraios/serena');
    expect(linkOf('docs/HARNESSES.md s0')).toBeNull();
    const html = render(e({ evidence: [{ ref: 'https://example.test/run', showed: 'x' }] }));
    expect(html).toContain('href="https://example.test/run"');
  });
  it('offers edit and delete, or restore for a deleted entry', () => {
    expect(render(base, { onEdit: () => {}, onDelete: () => {} })).toMatch(/\[ EDIT \][^]*\[ DELETE \]/);
    const del = render(e({ deleted: true, deleted_at: 1790700100, deleted_by: 'operator:x', deleted_why: 'old' }), { onRestore: () => {} });
    expect(del).toContain('DELETED');
    expect(del).toContain('[ RESTORE ]');
    expect(del).toContain('old');
  });
});

describe('EntryForm', () => {
  const render = (form = emptyForm('pi')) =>
    renderToStaticMarkup(createElement(EntryForm, { form, vocab, onChange: () => {}, onSubmit: () => {}, onCancel: () => {} }));
  it('has the harness, kind, status, proof and machine pickers, and says no key goes here', () => {
    const html = render();
    for (const label of ['HARNESS', 'KIND', 'STATUS', 'PROOF', 'APPLIES ON', 'EVIDENCE']) expect(html).toContain(label);
    expect(html).toContain('Never put a key here');
    expect(html).toContain('&lt;PASTE-YOUR-YAMADORI-KEY-HERE&gt;');
  });
  it('shows the download block for a download, and the skill folder for a skill', () => {
    expect(render({ ...emptyForm('pi'), kind: 'download' })).toContain('EXACT VERSION');
    expect(render({ ...emptyForm('pi'), kind: 'skill' })).toContain('SKILL FOLDER');
    expect(render()).not.toContain('EXACT VERSION');
  });
  it('the submit is disabled until the form is complete', () => {
    expect(render()).toMatch(/<button type="submit" disabled=""/);
  });
});

describe('route', () => {
  it('/harness is the HARNESS TOOLS screen', () => {
    expect(resolve((p) => ({ pattern: new RegExp(`^${p.replace(/:[^/]+/g, '([^/]+)')}$`), keys: [] }), '/harness')).toEqual({ name: 'harness' });
  });
  it('the kit downloads under a name that says which harness', () => {
    expect(kitFileName('codex')).toBe('yamadori-kit-codex.zip');
  });
});
