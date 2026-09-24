// The skills API (mcp/dash_skills.py): GET /dash/api/skills,
// GET /dash/api/skills/<id>, and the POSTs under /dash/api/skill. Every
// path is behind the same key as the rest of /dash/api.
import type { Tone } from '../ui/primitives';

export const SKILLS_PATH = '/dash/api/skills';
export const skillPath = (id: string) => `/dash/api/skills/${encodeURIComponent(id)}`;
export const SKILL_POST = {
  create: '/dash/api/skill',
  edit: '/dash/api/skill/edit',
  disable: '/dash/api/skill/disable',
  enable: '/dash/api/skill/enable',
  watch: '/dash/api/skill/watch',
  refetch: '/dash/api/skill/refetch',
} as const;

export type SkillStatus = 'pipeline' | 'armed' | 'quarantined' | 'failed' | 'disabled';

export type AppliesTo = { artifacts: string[]; languages: string[]; frameworks: string[] };

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
  text: string | null;
  text_version: number | null;
  applies_when: string | null;
  applies_to: AppliesTo | null;
  triggers: string[];
  meta: Record<string, unknown>;
};

export type Finding = { rule: string; action: string; what: string; line?: number; excerpt?: string };

export type SkillVersion = {
  version: number;
  origin: string;
  author: string | null;
  stage: string;
  state: string;
  reason: string | null;
  source_sha256: string | null;
  source_bytes: number | null;
  screen: { deterministic?: { ok: boolean; quarantine: Finding[]; notes: Finding[] }; model?: { ok?: boolean; skipped?: string } };
  validate: { counts?: Record<string, number>; dropped?: { item: string; why: string }[]; notes?: string[] };
  text: string | null;
  created: number;
  armed_at: number | null;
};

export type SkillDetail = SkillSummary & {
  versions: SkillVersion[];
  learned_triggers: string[];
  jobs: { id: string; queue: string; state: string; error: string | null; progress: string | null }[];
};

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
  counts: Record<SkillStatus, number>;
  recall: 'hints' | 'skills';
  limits: Record<string, unknown> & { label: string };
  thresholds: { emb_high: number; emb_low: number; label: string };
  learning: { rate: RateDay[]; pending: number; idle: { idle: boolean; why: string } };
};

export function statusTone(s: SkillStatus): Tone {
  switch (s) {
    case 'armed':
      return 'moss';
    case 'quarantined':
    case 'failed':
      return 'crimson';
    case 'disabled':
      return 'muted';
    default:
      return 'cyan';
  }
}

/** "v2 of 3" -- the served version against the newest one. */
export function versionText(s: Pick<SkillSummary, 'served_version' | 'latest_version'>): string {
  return s.served_version === null ? `none of ${s.latest_version}` : `v${s.served_version} of ${s.latest_version}`;
}

/** The body a create sends: a URL when the input is one http(s) URL, else the text. */
export function createBody(input: string): { url: string } | { text: string } {
  const t = input.trim();
  return /^https?:\/\/\S+$/.test(t) ? { url: t } : { text: input };
}

/** The whole period's fallback rate, or null when nothing had a candidate. */
export function overallFallbackRate(days: RateDay[]): number | null {
  const cand = days.reduce((a, d) => a + d.with_candidates, 0);
  return cand ? days.reduce((a, d) => a + d.fallbacks, 0) / cand : null;
}

export function appliesToText(a: AppliesTo | null): string {
  if (!a) return '—';
  const parts = [...a.artifacts, ...a.frameworks, ...a.languages];
  return parts.length ? parts.join(', ') : '—';
}
