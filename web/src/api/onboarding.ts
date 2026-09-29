// Package onboarding (mcp/onboarding.py behind mcp/dash_skills.py; the
// design is docs/PACKAGE-ONBOARDING.md section 8): a PROMPT WITH LINKS becomes
// one dataset of kind `package` that walks its stages -- resolve, clarify,
// index, vocab, examples, knn, sources, the skills JOIN, retire, the rebuild
// JOIN, evaluate. Every stage name, blocker, remedy and number the screens
// show comes from these payloads; nothing here hard-codes one.
import type { Tone } from '../ui/primitives';
import { URL_RE } from './skills';

export const ONBOARDING_PATH = '/dash/api/skill-factory/onboarding';
export const onboardingPath = (id: string) => `${ONBOARDING_PATH}/${encodeURIComponent(id)}`;
export const ONBOARDING_POST = {
  review: '/dash/api/skill-factory/onboarding/review',
  promote: '/dash/api/skill-factory/onboarding/promote',
  tier3: '/dash/api/skill-factory/onboarding/tier3',
} as const;
/** The datasets API's routes an onboarding reuses (mcp/dash_data.py). */
export const DATASET_POST = {
  answer: '/dash/api/dataset/answer',
  rerun: '/dash/api/dataset/rerun',
  advance: '/dash/api/dataset/advance',
} as const;

// ---------------------------------------------------------------------------
// Shapes (onboarding.summary / listing / detail)
// ---------------------------------------------------------------------------
export type OnboardingState = 'errored' | 'complete' | 'running' | 'waiting' | 'queued';
export type JobCounts = { queued?: number; running?: number; done?: number; errored?: number; cancelled?: number; total?: number };

export type OnboardingSummary = {
  id: string;
  group: string | null;
  package: string | null;
  version: string | null;
  stage: string;
  state: OnboardingState | string;
  updated: number;
  created: number;
  counts: Record<string, unknown>;
  jobs: JobCounts;
};

export type OnboardingJob = {
  id: string;
  queue: string;
  lane: string | null;
  state: string;
  stage: string | null;
  attempts: number | null;
  max_attempts: number | null;
  error: string | null;
  progress: string | null;
  created: number | null;
  started: number | null;
  finished: number | null;
  worker: string | null;
  not_before: number | null;
  parent: string | null;
  result?: unknown;
};

export type Blocker = { what: string; why?: string | null; retryable?: boolean | null; remedy?: string | null; owner?: string | null; field?: string | null; label?: string };

export type LicenceQuote = { spdx: string; quote: string; where?: string | null; kind?: string | null };

export type ResolvedPackage = {
  ecosystem?: string;
  name: string;
  version?: string | null;
  version_rule?: string | null;
  version_why?: string | null;
  commit?: string | null;
  commit_rule?: string | null;
  repository?: { owner: string; repo: string; directory?: string | null } | null;
  tarball?: string | null;
  integrity?: string | null;
  published?: string | null;
  licence_quotes?: LicenceQuote[];
  links?: string[];
  unsupported?: string;
  [k: string]: unknown;
};

export type Resolution = {
  links?: string[];
  package?: ResolvedPackage | null;
  sources?: { kind?: string; url?: string; path?: string; attach_to?: string | null; [k: string]: unknown }[];
  unresolved?: { link?: string | null; name?: string; why?: string | null }[];
  how?: string;
  group?: string | null;
  group_size?: number;
  siblings?: { id: string; package: string | null }[];
  replaces?: { mode: string; old?: string[]; why?: string; risk?: string } | null;
};

export type VocabRecord = {
  state?: 'promoted' | 'held' | 'forced' | 'skipped' | null;
  why?: string;
  package?: string;
  diff?: { per_package?: Record<string, Record<string, unknown>>; lost?: Record<string, string[]>; gained?: Record<string, string[]> } | null;
  floor?: { passed?: boolean; rows?: number; lost_tp?: { id: string; package: string }[]; gained_fp?: { id: string; package: string }[]; labels?: string } | null;
  forced_by?: string;
  forced_at?: number;
  [k: string]: unknown;
};

export type SourcesRecord = {
  chosen?: { tier: number; url?: string; path?: string; frontier?: boolean; why?: string }[];
  skipped?: { url?: string; path?: string; why?: string; tier?: number }[];
  tier3?: { where?: string; pages?: { url: string; title?: string }[]; ingest?: boolean; ingested_by?: string; ingested_at?: number; note?: string } | null;
  examples_dirs?: string[];
  made?: { created?: unknown[]; repointed?: unknown[]; unchanged?: unknown[]; existing?: unknown[]; failed?: unknown[]; to_retire?: unknown[] };
  [k: string]: unknown;
};

export type OnboardingSkill = {
  id: string;
  name: string;
  status: string;
  version: number | null;
  latest_version: number | null;
  reason: string | null;
  lead_for: string | null;
  parent: string | null;
  source_url: string | null;
};

export type Review = { stage: string; note: string; by: string; at: number };

export type OnboardingDetail = OnboardingSummary & {
  prompt: string | null;
  name: string | null;
  licence_value: string | null;
  stages: string[];
  joins: string[];
  job_rows: OnboardingJob[];
  blockers: Blocker[];
  missing: Blocker[];
  warnings: { kind: string; what: string }[];
  next_stage: string | null;
  field_states: unknown[];
  held: string | null;
  waiting: string | null;
  notes: { links?: string[]; aliases?: string[]; replaces?: string | null; author?: string; group?: string | null; [k: string]: unknown };
  resolution: Resolution | null;
  licence: { quotes?: LicenceQuote[]; chosen?: LicenceQuote | null; agree?: boolean; [k: string]: unknown } | null;
  index: { fetch?: Record<string, unknown>; health?: Record<string, unknown>; skipped?: string } | null;
  vocab: VocabRecord | null;
  examples: Record<string, unknown> | null;
  knn: { k?: number; k_n?: number; curve?: unknown; groups?: number; skipped?: string; [k: string]: unknown } | null;
  sources: SourcesRecord | null;
  skills: OnboardingSkill[];
  skill_counts: Record<string, number>;
  retire: { archived?: { id: string; name?: string; why?: string }[]; kept?: { id: string; why?: string }[]; vocab?: unknown; at?: number } | null;
  eval: Record<string, unknown> | null;
  reviews: Review[];
};

// ---------------------------------------------------------------------------
// CREATE: the PROMPT + LINKS body
// ---------------------------------------------------------------------------
export type ReplacesChoice = 'auto' | 'replace' | 'alongside';
export type PromptFields = { prompt: string; links: string; aliases: string; replaces: ReplacesChoice };

/** Split a free-text list on commas, newlines and (for links) whitespace. */
export function splitList(v: string, spaces = false): string[] {
  return v
    .split(spaces ? /[\s,]+/ : /[,\n]/)
    .map((x) => x.trim())
    .filter(Boolean);
}

const PROMPT_URL = /https?:\/\/[^\s<>"')\]]+/g;

/** The links the server will read: those in the prompt, then the field's (package_resolve.links_of). */
export function linksOf(prompt: string, links: string[]): string[] {
  const out: string[] = [];
  for (const raw of [...(prompt.match(PROMPT_URL) ?? []), ...links]) {
    const u = raw.trim().replace(/[.,;:!?]+$/, '');
    if (u && /^https?:\/\//.test(u) && !out.includes(u)) out.push(u);
  }
  return out;
}

/**
 * The POST /dash/api/skill body for a prompt with links, or the reason it
 * cannot be sent. `auto` leaves `replaces` out: the server's rule decides
 * (the same major replaces, a new major sits alongside).
 */
export function onboardingBody(f: PromptFields): { body: Record<string, unknown> } | { error: string } {
  const prompt = f.prompt.trim();
  if (!prompt) return { error: 'write the prompt: what to add, and its link(s)' };
  const links = splitList(f.links, true);
  const bad = links.find((u) => !URL_RE.test(u));
  if (bad) return { error: `${bad.slice(0, 80)} is not an http(s) URL` };
  if (!linksOf(prompt, links).length) return { error: 'give at least one link (npm, PyPI, GitHub, a SKILL.md or a docs page), in the prompt or the links field' };
  const aliases = splitList(f.aliases);
  const badAlias = aliases.find((a) => a.includes('=') && (!a.split('=')[0]?.trim() || !a.split('=').slice(1).join('=').trim()));
  if (badAlias) return { error: `${badAlias}: an alias binds as alias=package, both sides given` };
  const body: Record<string, unknown> = { prompt };
  if (links.length) body.links = links;
  if (aliases.length) body.aliases = aliases;
  if (f.replaces !== 'auto') body.replaces = f.replaces;
  return { body };
}

// ---------------------------------------------------------------------------
// The stage rail, from the job rows and the dataset's own stage
// ---------------------------------------------------------------------------
export type RailState = 'done' | 'running' | 'queued' | 'waiting' | 'failed' | 'held' | 'pending';
export type RailCell = { stage: string; state: RailState; job: OnboardingJob | null; join: boolean };

/** The newest job row of a stage (rows carry `created`; a re-run keeps its id). */
export function latestJob(rows: OnboardingJob[], stage: string): OnboardingJob | null {
  let best: OnboardingJob | null = null;
  for (const j of rows) {
    if (j.stage !== stage) continue;
    if (!best || (j.created ?? 0) >= (best.created ?? 0)) best = j;
  }
  return best;
}

function jobState(j: OnboardingJob, now: number): RailState {
  switch (j.state) {
    case 'done':
      return 'done';
    case 'running':
      return 'running';
    case 'errored':
    case 'cancelled':
      return 'failed';
    case 'queued':
      return j.not_before && j.not_before > now ? 'waiting' : 'queued';
    default:
      return 'queued';
  }
}

/**
 * One cell per stage. A stage with job rows takes its newest job's state;
 * a stage with none (submitted, clarify, the JOINS skills and rebuild,
 * complete) is drawn from where the dataset stands: done before it, pending
 * after it, and at it `held` when a person must act (a blocker owned by the
 * operator, a question field, clarify held for a person), `waiting` when the
 * worker owns what blocks it, `running` when nothing does. The job stage
 * the dataset stands at whose job is done but which cannot leave (a HELD
 * vocabulary) is `held`. `now` is epoch seconds.
 */
export function railCells(d: Pick<OnboardingDetail, 'stages' | 'joins' | 'stage' | 'job_rows' | 'blockers' | 'held'>, now: number): RailCell[] {
  const cur = d.stages.indexOf(d.stage);
  const joins = new Set(d.joins);
  const personHeld = !!d.held && d.stage === 'clarify';
  return d.stages.map((st, i) => {
    const job = latestJob(d.job_rows, st);
    const join = joins.has(st);
    const at = i === cur;
    if (job) {
      const s = jobState(job, now);
      if (at && s === 'done' && d.blockers.length) return { stage: st, state: 'held', job, join };
      if (cur >= 0 && i > cur && s === 'done') return { stage: st, state: 'pending', job, join };
      return { stage: st, state: s, job, join };
    }
    if (cur < 0) return { stage: st, state: 'pending', job: null, join };
    if (i < cur) return { stage: st, state: 'done', job: null, join };
    if (i > cur) return { stage: st, state: 'pending', job: null, join };
    if (st === 'complete') return { stage: st, state: 'done', job: null, join };
    if (personHeld || d.blockers.some((b) => b.owner === 'operator' || !!b.field)) return { stage: st, state: 'held', job: null, join };
    return { stage: st, state: d.blockers.length ? 'waiting' : 'running', job: null, join };
  });
}

export const RAIL_TONE: Record<RailState, Tone | null> = {
  done: 'moss',
  running: 'cyan',
  queued: 'cyan',
  waiting: 'cyan',
  failed: 'crimson',
  held: 'rose',
  pending: null,
};

export function onboardingTone(state: string): Tone {
  switch (state) {
    case 'complete':
      return 'moss';
    case 'errored':
      return 'crimson';
    case 'waiting':
      return 'rose';
    default:
      return 'cyan';
  }
}

/** "koota@0.6.6", or what is known before resolve answered. */
export function packageLabel(x: Pick<OnboardingSummary, 'package' | 'version' | 'id'>): string {
  if (!x.package) return `onboarding ${x.id}`;
  return x.version ? `${x.package}@${x.version}` : x.package;
}

// ---------------------------------------------------------------------------
// What the card and the page must offer a person
// ---------------------------------------------------------------------------

/** The waiting reason, only while a queued job is deferred (onboarding._waiting_reason). */
export function waitingLine(d: Pick<OnboardingDetail, 'waiting' | 'stage'>): string | null {
  const w = (d.waiting ?? '').trim();
  return w && d.stage !== 'complete' ? w : null;
}

/**
 * What clarify holds for. `licence`: the licence is unanswered, so the form
 * asks for the operator's statement (POST /dash/api/dataset/answer). `advance`:
 * a found licence carries a restriction and a person decides to serve it
 * (POST /dash/api/dataset/advance); the text is the server's reason.
 */
export function clarifyNeeds(d: Pick<OnboardingDetail, 'stage' | 'missing' | 'blockers' | 'held'>): { licence: Blocker | null; advance: string | null } {
  if (d.stage !== 'clarify') return { licence: null, advance: null };
  const lic = d.blockers.find((b) => b.field === 'licence') ?? d.missing.find((b) => b.field === 'licence') ?? null;
  return { licence: lic, advance: d.held ?? null };
}

/** The job a blocker names as errored, so its re-run is one click (datasets.rerun). */
export function erroredJobOf(b: Pick<Blocker, 'what'>): string | null {
  const m = /\bjob (\S+) is errored\b/.exec(b.what);
  return m?.[1] ?? null;
}

/** Errored job rows, newest per stage: the ones a person may re-run. */
export function rerunnableJobs(rows: OnboardingJob[]): OnboardingJob[] {
  const seen = new Set<string>();
  return rows
    .filter((j) => j.state === 'errored')
    .sort((a, b) => (b.created ?? 0) - (a.created ?? 0))
    .filter((j) => {
      const k = j.stage ?? j.id;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
}

/** A HELD vocabulary is the only state the FORCE button may act on (onboarding.promote). */
export const canForce = (v: VocabRecord | null | undefined): boolean => v?.state === 'held';

/** Recorded llms.txt pages that nobody has ingested yet (onboarding.tier3). */
export function tier3Pending(s: SourcesRecord | null | undefined): number {
  const t = s?.tier3;
  if (!t?.pages?.length || t.ingested_by) return 0;
  return t.pages.length;
}

/** The names each existing package loses under the new vocabulary, most first. */
export function lostNames(v: VocabRecord | null | undefined): { package: string; names: string[] }[] {
  return Object.entries(v?.diff?.lost ?? {})
    .filter(([, names]) => names?.length)
    .map(([p, names]) => ({ package: p, names }))
    .sort((a, b) => b.names.length - a.names.length || a.package.localeCompare(b.package));
}

// ---------------------------------------------------------------------------
// Generic readers for records whose shape the server owns (knn curve, eval)
// ---------------------------------------------------------------------------
export type CurvePoint = { k: string; value: string; n: number | null };

const fmt = (v: unknown): string => {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  return JSON.stringify(v);
};

/** k's curve, as rows: a list of {k, ...} objects, a {k: value} map, or a list of values (k from 1). */
export function curveRows(curve: unknown): CurvePoint[] {
  if (Array.isArray(curve)) {
    return curve.map((p, i) => {
      if (p && typeof p === 'object' && !Array.isArray(p)) {
        const o = p as Record<string, unknown>;
        const k = o.k ?? i + 1;
        const val = o.accuracy ?? o.top1 ?? o.value ?? o.score ?? Object.entries(o).find(([key, v]) => key !== 'k' && key !== 'n' && typeof v === 'number')?.[1];
        return { k: fmt(k), value: fmt(val), n: typeof o.n === 'number' ? o.n : null };
      }
      return { k: String(i + 1), value: fmt(p), n: null };
    });
  }
  if (curve && typeof curve === 'object') {
    return Object.entries(curve as Record<string, unknown>).map(([k, v]) => {
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        const o = v as Record<string, unknown>;
        return { k, value: fmt(o.accuracy ?? o.top1 ?? o.value), n: typeof o.n === 'number' ? o.n : null };
      }
      return { k, value: fmt(v), n: null };
    });
  }
  return [];
}

export type EvalRow = { where: string; measure: string; value: string; n: number | null; nFrom: string | null; num: boolean };
export type EvalView = { inSample: boolean | null; inSampleWhy: string | null; rows: EvalRow[] };

const N_KEYS = ['n', 'n_items', 'held_out_n', 'n_held_out', 'items'];

function nOf(o: Record<string, unknown>): number | null {
  for (const k of N_KEYS) if (typeof o[k] === 'number') return o[k] as number;
  return null;
}

/**
 * An evaluation record flattened for display: one row per scalar, each with
 * the n of its own object or of the nearest object above it that states one
 * (`nFrom` says which). A number with no n anywhere above it says so rather
 * than borrowing one. `in_sample` anywhere is lifted to the top.
 */
export function evalView(ev: Record<string, unknown> | null | undefined): EvalView {
  const out: EvalView = { inSample: null, inSampleWhy: null, rows: [] };
  if (!ev) return out;
  const walk = (o: Record<string, unknown>, where: string, inherited: { n: number | null; from: string | null }) => {
    const own = nOf(o);
    const nn = own !== null ? { n: own, from: where || '(top)' } : inherited;
    for (const [k, v] of Object.entries(o)) {
      const path = where ? `${where}.${k}` : k;
      if (k === 'in_sample') {
        if (typeof v === 'boolean') out.inSample = out.inSample === true ? true : v;
        else if (v && typeof v === 'object') {
          const iv = v as Record<string, unknown>;
          if (typeof iv.value === 'boolean') out.inSample = iv.value;
          if (typeof iv.why === 'string') out.inSampleWhy = iv.why;
          if (typeof iv.change === 'string') out.inSampleWhy = iv.change;
        }
        continue;
      }
      if ((k === 'in_sample_why' || k === 'in_sample_change') && typeof v === 'string') {
        out.inSampleWhy = v;
        continue;
      }
      if (N_KEYS.includes(k) && typeof v === 'number') continue; // shown as the n column
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        walk(v as Record<string, unknown>, path, nn);
      } else if (Array.isArray(v) && v.some((x) => x && typeof x === 'object' && !Array.isArray(x))) {
        v.forEach((x, i) => {
          if (x && typeof x === 'object' && !Array.isArray(x)) {
            const xo = x as Record<string, unknown>;
            const label = typeof xo.package === 'string' ? xo.package : typeof xo.name === 'string' ? xo.name : typeof xo.id === 'string' ? xo.id : String(i);
            walk(xo, `${path}[${label}]`, nn);
          }
        });
      } else {
        out.rows.push({ where: where || '(top)', measure: k, value: Array.isArray(v) ? v.map(fmt).join(', ') || '—' : fmt(v), n: nn.n, nFrom: nn.from, num: typeof v === 'number' });
      }
    }
  };
  walk(ev, '', { n: null, from: null });
  return out;
}

/** A record's scalar fields as [key, text] (nested values are left to a raw view). */
export function scalars(o: Record<string, unknown> | null | undefined, skip: string[] = []): [string, string][] {
  if (!o) return [];
  return Object.entries(o)
    .filter(([k, v]) => !skip.includes(k) && (v === null || ['string', 'number', 'boolean'].includes(typeof v)))
    .map(([k, v]) => [k, fmt(v)]);
}

/** The stage record a panel shows (null until the stage has run). */
export function stageRecord(d: OnboardingDetail, stage: string): unknown {
  switch (stage) {
    case 'resolve':
      return d.resolution;
    case 'clarify':
      return d.licence;
    case 'index':
      return d.index;
    case 'vocab':
      return d.vocab;
    case 'examples':
      return d.examples;
    case 'knn':
      return d.knn;
    case 'sources':
      return d.sources;
    case 'skills':
      return d.skills.length ? d.skills : null;
    case 'retire':
      return d.retire;
    case 'evaluate':
      return d.eval;
    default:
      return null;
  }
}
