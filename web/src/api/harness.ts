// HARNESS TOOLS: the harness kit entries (mcp/harness_kit.py) and the
// per-harness export for the operator's own machine (mcp/dash_harness.py).
// Every write goes through the keyed POST routes; the key is never part of
// an entry or a kit.
import { readKey } from './client';

export const HARNESS_KIT_PATH = '/dash/api/harness-kit';
export const HARNESS_DELETED_PATH = '/dash/api/harness-kit/deleted';
export const entryPath = (id: string) => `/dash/api/harness-kit/entry/${encodeURIComponent(id)}`;
export const exportPath = (harness: string) => `/dash/api/harness-kit/export/${encodeURIComponent(harness)}`;
export const POST = {
  create: '/dash/api/harness-kit/entry',
  edit: '/dash/api/harness-kit/entry/edit',
  del: '/dash/api/harness-kit/entry/delete',
  restore: '/dash/api/harness-kit/entry/restore',
} as const;

export type Status = 'recommended' | 'trying' | 'rejected';
export type Proof = 'operator' | 'measured' | 'verified' | 'unmeasured';
export type Where = 'remote' | 'box' | 'both';
export type Evidence = { ref: string; showed: string };
export type Download = {
  source_url: string;
  version: string;
  hash: string;
  commit: string;
  licence: string;
  pin_missing_why: string;
  verify: string;
  install: { windows?: string; unix?: string; any?: string };
};

export type Entry = {
  id: string;
  seed_id: string | null;
  version: number;
  harness: string;
  kind: string;
  title: string;
  body: string;
  evidence: Evidence[];
  status: Status;
  status_why: string;
  proof: Proof;
  where: Where;
  needs_operator: boolean;
  target: string;
  file_name: string;
  skill_path: string;
  download: Download | null;
  created: number;
  created_by: string;
  updated: number;
  updated_by: string;
  deleted: boolean;
  deleted_at: number | null;
  deleted_by: string | null;
  deleted_why: string | null;
  history?: { version: number; at: number; author: string; action: string; note: string | null; data: Partial<Entry> }[];
};

export type Labelled = { id: string; label: string };
export type KitOverview = {
  ok: true;
  entries: Entry[];
  counts: { total: number; by_harness: Record<string, number>; by_kind: Record<string, number>; by_status: Record<string, number>; needs_operator: number };
  public_base: { url: string; source: string };
  harnesses: Labelled[];
  kinds: Labelled[];
  statuses: Status[];
  proofs: Labelled[];
  wheres: Labelled[];
  skill_roots: string[];
  tokens: string[];
  key_placeholder: string;
};

/** The dropdown's special values beside a harness id. */
export const ALL = '*';

export type Filter = { harness: string; status: Status | ''; where: Where | ''; needsOperator: boolean; q: string };

/** What the page shows for a filter. A harness shows its own entries AND the
 * general ("any harness") ones, as the kit export does; "any" shows only the
 * general ones; ALL shows every entry. */
export function visible(entries: Entry[], f: Filter): Entry[] {
  const q = f.q.trim().toLowerCase();
  return entries.filter((e) => {
    if (f.harness !== ALL && !(e.harness === f.harness || (f.harness !== 'any' && e.harness === 'any'))) return false;
    if (f.status && e.status !== f.status) return false;
    if (f.where && !(e.where === f.where || (f.where !== 'both' && e.where === 'both'))) return false;
    if (f.needsOperator && !e.needs_operator) return false;
    if (q && !`${e.title}\n${e.body}\n${e.status_why}\n${e.evidence.map((x) => `${x.ref} ${x.showed}`).join('\n')}`.toLowerCase().includes(q)) return false;
    return true;
  });
}

const STATUS_ORDER: Record<string, number> = { recommended: 0, trying: 1, rejected: 2 };

/** Entries grouped by kind, in the vocabulary's order; inside a kind the
 * operator's pending decisions first, then by status, then title. */
export function byKind(entries: Entry[], kinds: Labelled[]): { kind: Labelled; entries: Entry[] }[] {
  return kinds
    .map((kind) => ({
      kind,
      entries: entries
        .filter((e) => e.kind === kind.id)
        .sort(
          (a, b) =>
            Number(b.needs_operator) - Number(a.needs_operator) ||
            (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9) ||
            a.title.localeCompare(b.title),
        ),
    }))
    .filter((g) => g.entries.length > 0);
}

/** An evidence ref is a link when it is a URL; a doc path or run name is text. */
export function linkOf(ref: string): string | null {
  const m = ref.match(/https?:\/\/[^\s)]+/);
  return m ? m[0] : null;
}

// ---------------------------------------------------------------- the form

export type Form = {
  harness: string;
  kind: string;
  title: string;
  body: string;
  status: Status;
  status_why: string;
  proof: Proof;
  where: Where;
  needs_operator: boolean;
  target: string;
  file_name: string;
  skill_path: string;
  evidence: Evidence[];
  dl: { source_url: string; version: string; hash: string; commit: string; licence: string; pin_missing_why: string; verify: string; windows: string; unix: string; any: string };
};

export function emptyForm(harness: string): Form {
  return {
    harness: harness === ALL ? 'any' : harness,
    kind: 'note',
    title: '',
    body: '',
    status: 'trying',
    status_why: '',
    proof: 'unmeasured',
    where: 'both',
    needs_operator: false,
    target: '',
    file_name: '',
    skill_path: '',
    evidence: [{ ref: '', showed: '' }],
    dl: { source_url: '', version: '', hash: '', commit: '', licence: '', pin_missing_why: '', verify: '', windows: '', unix: '', any: '' },
  };
}

export function formOf(e: Entry): Form {
  const d = e.download;
  return {
    harness: e.harness,
    kind: e.kind,
    title: e.title,
    body: e.body,
    status: e.status,
    status_why: e.status_why,
    proof: e.proof,
    where: e.where,
    needs_operator: e.needs_operator,
    target: e.target,
    file_name: e.file_name,
    skill_path: e.skill_path,
    evidence: e.evidence.length ? e.evidence.map((x) => ({ ...x })) : [{ ref: '', showed: '' }],
    dl: {
      source_url: d?.source_url ?? '',
      version: d?.version ?? '',
      hash: d?.hash ?? '',
      commit: d?.commit ?? '',
      licence: d?.licence ?? '',
      pin_missing_why: d?.pin_missing_why ?? '',
      verify: d?.verify ?? '',
      windows: d?.install?.windows ?? '',
      unix: d?.install?.unix ?? '',
      any: d?.install?.any ?? '',
    },
  };
}

/** The fields a form sends. The server validates; this only shapes. */
export function fieldsOf(f: Form): Record<string, unknown> {
  const dlUsed = Object.values(f.dl).some((v) => v.trim() !== '');
  const install: Record<string, string> = {};
  for (const k of ['windows', 'unix', 'any'] as const) if (f.dl[k].trim()) install[k] = f.dl[k].trim();
  return {
    harness: f.harness,
    kind: f.kind,
    title: f.title.trim(),
    body: f.body,
    status: f.status,
    status_why: f.status_why.trim(),
    proof: f.proof,
    where: f.where,
    needs_operator: f.needs_operator,
    target: f.target.trim(),
    file_name: f.file_name.trim(),
    skill_path: f.kind === 'skill' ? f.skill_path.trim() : '',
    evidence: f.evidence.filter((e) => e.ref.trim() || e.showed.trim()).map((e) => ({ ref: e.ref.trim(), showed: e.showed.trim() })),
    download: dlUsed
      ? {
          source_url: f.dl.source_url.trim(),
          version: f.dl.version.trim(),
          hash: f.dl.hash.trim(),
          commit: f.dl.commit.trim(),
          licence: f.dl.licence.trim(),
          pin_missing_why: f.dl.pin_missing_why.trim(),
          verify: f.dl.verify.trim(),
          install,
        }
      : null,
  };
}

/** Problems the page can name before sending (the server repeats every rule
 * and answers with its own reasons). */
export function formProblems(f: Form): string[] {
  const out: string[] = [];
  if (!f.title.trim()) out.push('a title');
  const ev = f.evidence.filter((e) => e.ref.trim() || e.showed.trim());
  if (f.status !== 'trying' && ev.length === 0) out.push(`evidence: a ${f.status} entry names what it rests on`);
  if (ev.some((e) => !e.ref.trim() || !e.showed.trim())) out.push('each evidence item needs a ref and what it showed');
  if (f.status !== 'recommended' && !f.status_why.trim()) out.push(`why it is ${f.status}`);
  if (f.kind === 'download' && !Object.values(f.dl).some((v) => v.trim())) out.push('the download: source, exact version, hash or commit, licence, install');
  if (f.kind === 'skill' && !f.skill_path.trim() && !f.body.trimStart().startsWith('---')) out.push('a skill folder path, or a SKILL.md text starting with ---');
  return out;
}

// ---------------------------------------------------------------- the kit

/** Fetch the harness's kit as a zip with this browser's key and hand it to
 * the browser as a download. Returns an error text, or null on success. */
export async function downloadKit(harness: string): Promise<string | null> {
  const key = readKey();
  if (!key) return 'no key in this browser';
  let res: Response;
  try {
    res = await fetch(exportPath(harness), { headers: { Authorization: `Bearer ${key}` }, cache: 'no-store' });
  } catch (e) {
    return `cannot reach ${exportPath(harness)}: ${String(e)}`;
  }
  if (!res.ok) {
    const t = await res.text();
    try {
      const j = JSON.parse(t) as { error?: string };
      return `${res.status}: ${j.error ?? t.slice(0, 200)}`;
    } catch {
      return `${res.status}: ${t.slice(0, 200)}`;
    }
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = kitFileName(harness);
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return null;
}

export function kitFileName(harness: string): string {
  return `yamadori-kit-${harness}.zip`;
}

export function when(ts: number | null | undefined): string {
  if (!ts) return '';
  return new Date(ts * 1000).toISOString().slice(0, 16).replace('T', ' ') + 'Z';
}
