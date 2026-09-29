// The skill factory API (mcp/dash_skills.py; the contract is
// docs/SKILL-FACTORY.md): GET /dash/api/skills, /dash/api/skills/<id>,
// /dash/api/skill-factory/{prompts,selections,recent}, and the POSTs under
// /dash/api/skill. Every path is behind the same key as the rest of
// /dash/api. Every cap, count, stage and taxonomy value the screens show
// comes from these payloads; nothing here hard-codes one.
import type { Tone } from '../ui/primitives';

export const SKILLS_PATH = '/dash/api/skills';
export const skillPath = (id: string) => `/dash/api/skills/${encodeURIComponent(id)}`;
export const skillMdPath = (id: string) => `${skillPath(id)}/skill.md`;
export const PROMPTS_PATH = '/dash/api/skill-factory/prompts';
export const SELECTIONS_PATH = '/dash/api/skill-factory/selections';
export const RECENT_PATH = '/dash/api/skill-factory/recent';
export const SKILL_POST = {
  create: '/dash/api/skill',
  edit: '/dash/api/skill/edit',
  tests: '/dash/api/skill/tests',
  activation: '/dash/api/skill/activation',
  disable: '/dash/api/skill/disable',
  enable: '/dash/api/skill/enable',
  archive: '/dash/api/skill/archive',
  quarantine: '/dash/api/skill/quarantine',
  rerun: '/dash/api/skill/rerun',
  licence: '/dash/api/skill/licence',
  watch: '/dash/api/skill/watch',
  refetch: '/dash/api/skill/refetch',
} as const;

export type SkillStatus = 'pipeline' | 'armed' | 'quarantined' | 'failed' | 'disabled' | 'archived' | 'decomposed';

export type AppliesTo = { artifacts: string[]; languages: string[]; frameworks: string[] };
export type Category = { artifact: string[]; language: string[]; framework: string[]; phase: string[]; domain: string[] };
export type Gates = {
  phases: string[];
  tools_any: string[];
  tools_all: string[];
  tools_none: string[];
  situations: string[];
  all_of: string[];
  topics: string[];
};
export type Licence = { spdx?: string | null; quote?: string | null; where?: string | null; by?: string; at?: number };
export type Provenance = {
  kind?: string;
  source?: string;
  url?: string;
  parent?: string;
  section?: string;
  licence?: Licence;
  fetched_at?: number;
  sha256?: string;
  [k: string]: unknown;
};
export type ActivationSummary = { passed: boolean | null; score: number | null; n: number | null; failures: unknown[] | null };
/** The injected body against the caps (dash_skills.size_of). Absent from a server older than the field. */
export type Size = { tokens: number; chars: number; items: number; prohibitions: number; longest_item_chars: number; description_chars: number };

export type SkillSummary = {
  id: string;
  name: string;
  title: string | null;
  source_url: string | null;
  source_kind: string;
  status: SkillStatus;
  reason: string | null;
  enabled: boolean;
  served_version: number | null;
  latest_version: number;
  watch_seconds: number | null;
  next_watch: number | null;
  created: number;
  updated: number;
  meta: Record<string, unknown>;
  text_version: number | null;
  description: string | null;
  tags: string[];
  provenance: Provenance | null;
  applies_when: string | null;
  applies_to: AppliesTo | null;
  category: Category | null;
  gates: Gates | null;
  triggers: string[];
  activation: ActivationSummary | null;
  folder: string | null;
  size?: Size | null;
};

export type TestCase = { text: string; files?: string[]; tool_output?: string; route_class?: string };
export type BehaviourCheck = { prompt: string; check?: { kind?: string; pattern?: string; [k: string]: unknown }; why?: string };
export type SkillTests = {
  activation?: { should?: TestCase[]; should_not?: TestCase[] };
  behaviour?: BehaviourCheck[];
  model?: unknown;
};
export type CaseResult = {
  kind: 'should' | 'should_not' | string;
  text: string;
  ok: boolean;
  verdict: string | null;
  strength: string | null;
  score: number | null;
  why: string[] | null;
  gate: string | null;
  rank?: number | null;
};
export type ActivationRun = {
  passed: boolean;
  score: number | null;
  n: number | null;
  failures: unknown[] | null;
  cases?: CaseResult[];
  pool?: { size?: number; should_within_cap?: unknown; should?: unknown };
  embedding?: string;
};
export type Item = { form: string; situation?: string; text: string; quote?: string; ref?: string; provenance?: string };
export type Finding = { rule: string; action: string; what: string; line?: number; excerpt?: string };
export type ValidateRecord = {
  ok?: boolean;
  title?: string;
  applies_when?: string;
  items?: Item[];
  dropped?: { item?: string; why: string }[];
  notes?: string[];
  counts?: Record<string, number>;
  why?: string;
  quarantine?: unknown[];
  activation?: ActivationRun;
  skill_md_problems?: string[];
};
export type SkillVersion = {
  version: number;
  origin: string;
  author: string | null;
  path: string[] | null;
  stage: string;
  state: string;
  reason: string | null;
  source_sha256: string | null;
  source_bytes: number | null;
  fetched: Record<string, unknown> | null;
  screen: { deterministic?: { ok: boolean; quarantine: Finding[]; notes: Finding[]; rules?: string[] }; model?: Record<string, unknown> } | null;
  classify: Record<string, unknown> | null;
  licence: (Licence & { searched?: string[] }) | null;
  tests: SkillTests | null;
  validate: ValidateRecord | null;
  distil: Record<string, unknown> | null;
  text: string | null;
  created: number;
  updated: number | null;
  armed_at: number | null;
  meta: Record<string, unknown> | null;
};
export type SkillJob = {
  id: string;
  queue: string;
  lane?: string;
  state: string;
  stage: string | null;
  attempts?: number;
  max_attempts?: number;
  error: string | null;
  progress: string | null;
  result?: unknown;
  created?: number;
  finished?: number | null;
};
export type Selection = {
  ts: number;
  skill: string;
  version: number | null;
  route_class: string | null;
  decided_by: string | null;
  strength: string | null;
  request: string | null;
};
export type SkillDetail = SkillSummary & {
  skill_md: string | null;
  text: string | null;
  versions: SkillVersion[];
  jobs: SkillJob[];
  tests: SkillTests;
  children: { id: string; name: string; status: SkillStatus }[];
  selections: Selection[];
  learned_triggers: string[];
};

export type TaxonomyEntry = string | { id: string; name: string };
export type Limits = {
  label: string;
  chars_per_token: number;
  skill_tokens_aim: [number, number];
  skill_tokens_hard: number;
  max_items: number;
  max_item_chars: number;
  max_prohibitions: number;
  name_chars: number;
  description_chars: number;
  max_skills_per_turn: number;
  [k: string]: unknown;
};
export type PromptMeta = { name: string; version: string; stage: string; chars: number; sha256: string; text?: string };

export type RateDay = {
  day: string;
  requests: number;
  with_candidates: number;
  injected: number;
  fallbacks: number;
  fallback_errors: number;
  cache_hits: number;
  fallback_rate: number | null;
};

export type SkillsOverview = {
  skills: SkillSummary[];
  counts: Record<string, number>;
  recall?: string;
  skip_classes?: string[];
  stages: string[];
  paths: Record<string, string[]>;
  taxonomy: Record<string, TaxonomyEntry[] | string>;
  taxonomy_counts?: Record<string, Record<string, number>>;
  limits: Limits;
  /** Empty since 2026-09-27: no cosine thresholds (a cosine only ranks). */
  thresholds?: Record<string, unknown>;
  prompts: PromptMeta[];
  learning?: { rate: RateDay[]; pending: number; idle: { idle: boolean; why: string } };
  queue?: { states?: Record<string, number>; lanes?: Record<string, Record<string, number>>; paused?: Record<string, unknown> };
};

export type RecentMatched = {
  id: string;
  version: number | null;
  name: string | null;
  title: string | null;
  decided_by: string | null;
  strength: string | null;
  cosine: number | null;
  confidence: number | null;
  why: string[] | null;
};
export type RecentSkillsRecord = {
  on?: boolean;
  route_class?: string | null;
  ids?: string[];
  versions?: number[];
  names?: (string | null)[];
  chars?: number;
  tokens?: number;
  why?: string;
  candidates?: number;
  armed?: number;
  cache?: string;
  replayed?: boolean;
  matched: RecentMatched[];
  dropped: { id: string; why: string }[];
  fallback: { ran?: boolean; ok?: boolean } | null;
  embedding: { ok?: boolean; why?: string } | null;
};
export type RecentRequests = { requests: { at: number; utility: boolean; skills: RecentSkillsRecord }[]; keep: number; limits: Limits };

// ---------------------------------------------------------------------------
// Pure helpers (Skills.test.ts)
// ---------------------------------------------------------------------------

export function statusTone(s: string): Tone {
  switch (s) {
    case 'armed':
      return 'moss';
    case 'quarantined':
    case 'failed':
      return 'crimson';
    case 'disabled':
    case 'archived':
    case 'superseded':
      return 'muted';
    default:
      return 'cyan';
  }
}

/** "v2 of 3" -- the served version against the newest one. */
export function versionText(s: Pick<SkillSummary, 'served_version' | 'latest_version'>): string {
  return s.served_version === null ? `none of ${s.latest_version}` : `v${s.served_version} of ${s.latest_version}`;
}

export function appliesToText(a: AppliesTo | null): string {
  if (!a) return '—';
  const parts = [...a.artifacts, ...a.frameworks, ...a.languages];
  return parts.length ? parts.join(', ') : '—';
}

/** The whole period's fallback rate, or null when nothing had a candidate. */
export function overallFallbackRate(days: RateDay[]): number | null {
  const cand = days.reduce((a, d) => a + d.with_candidates, 0);
  return cand ? days.reduce((a, d) => a + d.fallbacks, 0) / cand : null;
}

export const entryId = (e: TaxonomyEntry) => (typeof e === 'string' ? e : e.id);
export const entryName = (e: TaxonomyEntry) => (typeof e === 'string' ? e : e.name);

/** A facet: an axis the library filters on, and how to read it off a skill. */
export type Facet = { key: string; label: string; get: (s: SkillSummary) => string[]; names: Record<string, string> };

const uniq = (xs: (string | null | undefined)[]) => [...new Set(xs.filter((x): x is string => !!x))];

export function licenceOf(s: SkillSummary): string {
  const spdx = s.provenance?.licence?.spdx;
  return spdx ? String(spdx) : 'none recorded';
}

/**
 * The facets, from the overview's own taxonomy (names) plus every axis the
 * skill rows carry. Values not in the taxonomy (domains, tools) still show.
 */
export function facetsOf(o: Pick<SkillsOverview, 'taxonomy'>): Facet[] {
  const names = (axis: string): Record<string, string> => {
    const t = o.taxonomy[axis];
    return Array.isArray(t) ? Object.fromEntries(t.map((e) => [entryId(e), entryName(e)])) : {};
  };
  return [
    { key: 'status', label: 'state', get: (s) => [s.status], names: {} },
    { key: 'artifact', label: 'artifact', get: (s) => s.category?.artifact ?? [], names: names('artifact') },
    { key: 'language', label: 'language', get: (s) => s.category?.language ?? [], names: names('language') },
    { key: 'framework', label: 'framework', get: (s) => s.category?.framework ?? [], names: names('framework') },
    { key: 'phase', label: 'phase', get: (s) => s.category?.phase ?? [], names: names('phase') },
    { key: 'domain', label: 'domain', get: (s) => s.category?.domain ?? [], names: names('domain') },
    { key: 'situation', label: 'situation', get: (s) => s.gates?.situations ?? [], names: names('situation') },
    {
      key: 'tools',
      label: 'client tools',
      get: (s) =>
        uniq([
          ...(s.gates?.tools_any ?? []).map((t) => `any:${t}`),
          ...(s.gates?.tools_all ?? []).map((t) => `all:${t}`),
          ...(s.gates?.tools_none ?? []).map((t) => `none:${t}`),
        ]),
      names: {},
    },
    { key: 'kind', label: 'provenance', get: (s) => [s.provenance?.kind || s.source_kind], names: {} },
    { key: 'licence', label: 'licence', get: (s) => [licenceOf(s)], names: {} },
  ];
}

export type Filters = Record<string, string[]>;

/** Case-insensitive match over name, title, description, tags, topics, triggers and id. */
export function matchesQuery(s: SkillSummary, q: string): boolean {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = [s.id, s.name, s.title, s.description, s.applies_when, ...s.tags, ...(s.gates?.topics ?? []), ...s.triggers].filter(Boolean).join('\n').toLowerCase();
  return words.every((w) => hay.includes(w));
}

/** Within a facet any selected value matches (OR); across facets all must (AND). */
export function filterSkills(skills: SkillSummary[], facets: Facet[], filters: Filters, q: string, except?: string): SkillSummary[] {
  return skills.filter((s) => {
    for (const f of facets) {
      if (f.key === except) continue;
      const want = filters[f.key];
      if (want?.length && !f.get(s).some((v) => want.includes(v))) return false;
    }
    return matchesQuery(s, q);
  });
}

/** Each value's count among the skills the OTHER facets and the query leave: the usual facet count. */
export function facetCounts(skills: SkillSummary[], facets: Facet[], filters: Filters, q: string, facet: Facet): [string, number][] {
  const counts = new Map<string, number>();
  for (const s of filterSkills(skills, facets, filters, q, facet.key)) for (const v of facet.get(s)) counts.set(v, (counts.get(v) ?? 0) + 1);
  // Selected values stay listed even at zero, so they can be cleared.
  for (const v of filters[facet.key] ?? []) if (!counts.has(v)) counts.set(v, 0);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

export function toggle(filters: Filters, key: string, value: string): Filters {
  const cur = filters[key] ?? [];
  const next = cur.includes(value) ? cur.filter((v) => v !== value) : [...cur, value];
  const out = { ...filters };
  if (next.length) out[key] = next;
  else delete out[key];
  return out;
}

/** moss within the aim, cyan over the aim, crimson over the hard cap. */
export function sizeTone(tokens: number, l: Pick<Limits, 'skill_tokens_aim' | 'skill_tokens_hard'>): Tone {
  if (tokens > l.skill_tokens_hard) return 'crimson';
  if (tokens > l.skill_tokens_aim[1]) return 'cyan';
  return 'moss';
}

/** Every cap a skill's size breaks, as sentences. Empty when it fits. */
export function overCaps(sz: Size, l: Limits): string[] {
  const out: string[] = [];
  if (sz.tokens > l.skill_tokens_hard) out.push(`${sz.tokens} tokens, over the ${l.skill_tokens_hard}-token hard cap`);
  if (sz.items > l.max_items) out.push(`${sz.items} items, over ${l.max_items}`);
  if (sz.prohibitions > l.max_prohibitions) out.push(`${sz.prohibitions} prohibitions, over ${l.max_prohibitions}`);
  if (sz.longest_item_chars > l.max_item_chars) out.push(`an item of ${sz.longest_item_chars} characters, over ${l.max_item_chars}`);
  if (sz.description_chars > l.description_chars) out.push(`a description of ${sz.description_chars} characters, over ${l.description_chars}`);
  return out;
}

export type SortKey = 'title' | 'updated' | 'tokens' | 'score';
export function sortSkills(xs: SkillSummary[], by: SortKey): SkillSummary[] {
  const t = (s: SkillSummary) => (s.title || s.name).toLowerCase();
  const cmp: Record<SortKey, (a: SkillSummary, b: SkillSummary) => number> = {
    title: (a, b) => t(a).localeCompare(t(b)),
    updated: (a, b) => b.updated - a.updated,
    tokens: (a, b) => (b.size?.tokens ?? -1) - (a.size?.tokens ?? -1),
    score: (a, b) => (a.activation?.score ?? 2) - (b.activation?.score ?? 2),
  };
  return [...xs].sort((a, b) => cmp[by](a, b) || a.id.localeCompare(b.id));
}

/**
 * The situation and the remedy of a failure reason. The pipeline writes
 * "... Remedy (operator): ..." or worker.describe's "situation |
 * retryable: yes | remedy (owner): ...".
 */
export function splitRemedy(reason: string | null | undefined): { situation: string; retryable: string | null; remedy: string | null } {
  const r = (reason ?? '').trim();
  const m = /^(.*?)[\s.|]*\bremedy\s*(\([^)]*\))?\s*:\s*(.*)$/is.exec(r);
  if (!m) return { situation: r, retryable: null, remedy: null };
  let situation = (m[1] ?? '').trim();
  let retryable: string | null = null;
  const rt = /\|\s*retryable:\s*(\w+)\s*$/i.exec(situation);
  if (rt) {
    retryable = rt[1] ?? null;
    situation = situation.slice(0, rt.index).trim();
  }
  situation = situation.replace(/[|.\s]+$/, '');
  return { situation, retryable, remedy: `${m[2] ? `${m[2]} ` : ''}${(m[3] ?? '').trim()}` };
}

export type StageState = 'done' | 'running' | 'queued' | 'failed' | 'quarantined' | 'decomposed' | 'armed' | 'pending' | 'skipped';

/**
 * One version's walk through its path: every stage before the current one
 * is done; the current one carries the version's state; the rest wait.
 * Stages of the full list not on the path are `skipped`.
 */
export function stageStates(stages: string[], v: Pick<SkillVersion, 'path' | 'stage' | 'state'>, jobs: SkillJob[] = []): { stage: string; state: StageState }[] {
  const path = v.path ?? [];
  const cur = path.indexOf(v.stage);
  const job = (st: string) => jobs.find((j) => j.stage === st && (j.state === 'queued' || j.state === 'running'));
  return stages.map((st) => {
    const i = path.indexOf(st);
    if (i < 0) return { stage: st, state: 'skipped' as StageState };
    if (v.state === 'planned') return { stage: st, state: 'pending' as StageState };
    if (cur < 0 || i < cur) return { stage: st, state: 'done' as StageState };
    if (i > cur) return { stage: st, state: 'pending' as StageState };
    switch (v.state) {
      case 'armed':
      case 'superseded':
        return { stage: st, state: 'armed' as StageState };
      case 'failed':
        return { stage: st, state: 'failed' as StageState };
      case 'quarantined':
        return { stage: st, state: 'quarantined' as StageState };
      case 'decomposed':
        return { stage: st, state: 'decomposed' as StageState };
      default:
        return { stage: st, state: (job(st)?.state === 'queued' ? 'queued' : 'running') as StageState };
    }
  });
}

/** Which stages' outputs a version carries, keyed by stage. */
export function stageOutput(v: SkillVersion, stage: string): unknown {
  switch (stage) {
    case 'fetch':
      return v.fetched;
    case 'screen':
      return v.screen?.deterministic ?? null;
    case 'screen_model':
      return v.screen?.model ?? null;
    case 'licence':
      return v.licence && Object.keys(v.licence).length ? v.licence : null;
    case 'distil':
    case 'decompose':
      return v.distil && Object.keys(v.distil).length ? v.distil : null;
    case 'classify':
      return v.classify && Object.keys(v.classify).length ? v.classify : null;
    case 'tests':
      return v.tests && Object.keys(v.tests).length ? v.tests : null;
    case 'validate':
      return v.validate && Object.keys(v.validate).length ? v.validate : null;
    default:
      return null;
  }
}

/** A frontier SKILL.md: frontmatter with a name and a description (skills.is_skill_md). */
export function isSkillMd(text: string): boolean {
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text.trimStart());
  const fm = m?.[1] ?? '';
  return !!m && /^name\s*:\s*\S/m.test(fm) && /^description\s*:\s*\S/m.test(fm);
}

/** A URL the backend walks on the frontier path (skills.create). */
export function isFrontierUrl(u: string): boolean {
  return /\/SKILL\.md$/.test(u) || /github\.com\/.+\/tree\//.test(u);
}

export const URL_RE = /^https?:\/\/\S+$/;

export type CreateMode = 'text' | 'urls' | 'frontier';
export type CreateFields = { mode: CreateMode; input: string; goal: string; name: string; watchHours: string };

/** The POST /dash/api/skill body for the form, or the reason it cannot be sent. */
export function submitBody(f: CreateFields, maxBatch = 25): { body: Record<string, unknown> } | { error: string } {
  const extra: Record<string, unknown> = {};
  if (f.goal.trim()) extra.goal = f.goal.trim();
  if (f.name.trim()) extra.name = f.name.trim();
  if (f.mode === 'urls') {
    const urls = f.input.split(/\s+/).map((u) => u.trim()).filter(Boolean);
    if (!urls.length) return { error: 'give at least one URL' };
    const bad = urls.find((u) => !URL_RE.test(u));
    if (bad) return { error: `${bad.slice(0, 80)} is not an http(s) URL` };
    if (urls.length > maxBatch) return { error: `at most ${maxBatch} URLs at once` };
    if (f.watchHours.trim() !== '') {
      const h = Number(f.watchHours);
      if (!Number.isFinite(h) || h < 0) return { error: 'watch hours must be a number ≥ 0' };
      extra.watch_hours = h;
    }
    if (urls.length === 1) return { body: { url: urls[0], ...extra } };
    delete extra.name; // one name cannot name a batch
    return { body: { urls, ...extra } };
  }
  if (!f.input.trim()) return { error: 'paste the text' };
  if (f.mode === 'frontier') {
    if (!isSkillMd(f.input)) return { error: 'a frontier SKILL.md opens with frontmatter carrying a name and a description' };
    return { body: { text: f.input, frontier: true, ...extra } };
  }
  return { body: { text: f.input, ...extra } };
}

/** One decision: the selections recorded together for one request. */
export type Decision = { ts: number; request: string | null; route_class: string | null; rows: Selection[] };

export function groupDecisions(rows: Selection[]): Decision[] {
  const out = new Map<string, Decision>();
  for (const r of rows) {
    const k = `${r.ts}|${r.request ?? ''}`;
    const d = out.get(k) ?? { ts: r.ts, request: r.request, route_class: r.route_class, rows: [] };
    d.rows.push(r);
    out.set(k, d);
  }
  return [...out.values()].sort((a, b) => b.ts - a.ts);
}

/** The items of the version shown: its validate record, else nothing. */
export function itemsOf(d: Pick<SkillDetail, 'versions' | 'served_version' | 'text_version'>): Item[] {
  const v = d.versions.find((x) => x.version === (d.text_version ?? d.served_version)) ?? d.versions[0];
  return v?.validate?.items ?? [];
}

/** The SKILL.md split into its frontmatter and body. */
export function splitSkillMd(md: string): { frontmatter: string | null; body: string } {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(md);
  return m ? { frontmatter: m[1] ?? '', body: md.slice(m[0].length) } : { frontmatter: null, body: md };
}

/** A stage's rerun is allowed only for a version that is not armed, superseded or decomposed (skills.rerun). */
export function canRerun(v: Pick<SkillVersion, 'state'> | undefined): boolean {
  return !!v && !['armed', 'superseded', 'decomposed'].includes(v.state);
}
