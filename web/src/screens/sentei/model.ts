// Pure reads of /dash/api/results for the benchmark page: the headline strip
// and the mechanism-health rows. No statistic is computed here; every number
// is one the server already carries (mcp/dash_results.py), picked and
// labelled with its n.
import type {
  DomainMechCount,
  DomainTriggers,
  EmptySection,
  ErrorSection,
  LbArm,
  LbRun,
  LiveBenchSection,
  MechHealth,
  Results,
  SweArm,
} from '../../api/types';

export function isState(x: unknown, ...states: string[]): boolean {
  return !!x && typeof x === 'object' && states.includes(String((x as { state?: unknown }).state));
}

export const isEmpty = (x: unknown): x is EmptySection => isState(x, 'empty');
export const isError = (x: unknown): x is ErrorSection => isState(x, 'error');

export const f1 = (x: number | null | undefined): string => (x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(1));
export const f2 = (x: number | null | undefined): string => (x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(2));
export const pc0 = (x: number | null | undefined): string => (x === null || x === undefined || !Number.isFinite(x) ? '—' : `${Math.round(x * 100)}%`);

/** "70.5 [51.4–85.9]" for a 0..100 score with its interval. */
export function scoreCi(score: number | null | undefined, ci: [number, number] | null | undefined): string {
  if (score === null || score === undefined) return '—';
  return ci ? `${f1(score)} [${f1(ci[0])}–${f1(ci[1])}]` : f1(score);
}

/** "2/2 · 100% [34%–100%]" for k of n with a Wilson interval. */
export function kOfN(k: number, n: number, lo?: number | null, hi?: number | null): string {
  if (!n) return `${k}/0`;
  const ci = lo !== null && lo !== undefined && hi !== null && hi !== undefined ? ` [${pc0(lo)}–${pc0(hi)}]` : '';
  return `${k}/${n} · ${pc0(k / n)}${ci}`;
}

/** Questions scored in an arm: the sum of its categories' n. */
export const lbN = (a: LbArm): number => Object.values(a.categories ?? {}).reduce((s, c) => s + (c?.n ?? 0), 0);

export function currentRun(lb: LiveBenchSection): LbRun | null {
  const runs = Array.isArray(lb.runs) ? lb.runs : [];
  return runs.find((r) => r.run_id === lb.current) ?? runs.find((r) => !r.frozen) ?? null;
}

// ------------------------------------------------------------ mechanisms --

/** One mechanism for one arm. undefined = the source has no such count; null = not exposed on the response. */
export type MechRow = {
  arm: string;
  mechanism: string;
  n: number;
  unit: 'answers' | 'tasks' | 'turns';
  allowed?: number | null;
  decided?: number | null;
  ran?: number | null;
  produced?: number | null;
  extra?: string;
  /** denominator for ran / produced when it is not n (analyse.py: the rows that allowed it) */
  base?: number;
};

const MECH_LABEL: Record<string, string> = {
  retrieval: 'retrieval',
  hints: 'hints',
  deep_thinking: 'deep thinking',
  fanout: 'fan-out',
  check_code: 'check_code',
  repair: 'repair',
  self_check: 'self-check (client tool)',
};

/** bench/livebench/mechanisms.py health() -> rows. */
export function mechRowsLiveBench(arm: string, h: MechHealth | undefined): MechRow[] {
  const m = h?.mechanisms;
  if (!h || !m) return [];
  return Object.entries(m).map(([mech, c]) => {
    const extras = Object.entries(c)
      .filter(([k]) => !['allowed', 'decided', 'ran', 'produced'].includes(k))
      .map(([k, v]) => `${k.replace(':', ' ')} ${v ?? '—'}`);
    return {
      arm,
      mechanism: MECH_LABEL[mech] ?? mech,
      n: h.recorded ?? h.answers,
      unit: 'answers' as const,
      allowed: 'allowed' in c ? c.allowed : undefined,
      decided: 'decided' in c ? c.decided : undefined,
      ran: 'ran' in c ? c.ran : undefined,
      produced: 'produced' in c ? c.produced : undefined,
      extra: extras.join(' · ') || undefined,
    };
  });
}

/** bench/domain/analyse.py mechanism_health(): allowed of all rows; ran and produced of the rows that allowed it. */
export function mechRowsDomainHealth(arm: string, mh: Record<string, DomainMechCount> | undefined): MechRow[] {
  if (!mh) return [];
  return Object.entries(mh)
    .filter(([, c]) => c && c.rows > 0)
    .map(([mech, c]) => ({
      arm,
      mechanism: MECH_LABEL[mech] ?? mech.replace('_', ' '),
      n: c.rows,
      unit: 'tasks' as const,
      allowed: c.allowed,
      ran: c.ran,
      produced: c.data,
      base: c.allowed,
    }));
}

/** bench/domain/analyse.py triggers() -> rows (read off x_yamadori per task). */
export function mechRowsDomain(arm: string, t: DomainTriggers | undefined): MechRow[] {
  if (!t || !t.n) return [];
  const k = (r?: { k: number }) => (r ? r.k : undefined);
  return [
    { arm, mechanism: 'retrieval', n: t.n, unit: 'tasks', allowed: k(t.tools_offered), ran: k(t.main_hops_gt1), produced: null },
    { arm, mechanism: 'hints', n: t.n, unit: 'tasks', decided: k(t.hints_selected), produced: k(t.hints_emitted) },
    {
      arm, mechanism: 'deep thinking', n: t.n, unit: 'tasks',
      decided: k(t.investigate_chosen), ran: k(t.investigate_searched), produced: k(t.investigate_injected),
    },
    { arm, mechanism: 'fan-out', n: t.n, unit: 'tasks', decided: k(t.fanned_out) },
  ];
}

/** bench/swebench results.jsonl x_yamadori sums -> rows, per agent turn. */
export function mechRowsSwe(a: SweArm): MechRow[] {
  const m = a.mechanisms ?? {};
  const n = m.turns ?? 0;
  if (!n) return [];
  return [
    { arm: a.arm, mechanism: 'retrieval', n, unit: 'turns', allowed: m.tools_offered_turns, ran: m.internal_tool_turns, produced: null },
    { arm: a.arm, mechanism: 'hints', n, unit: 'turns', ran: m.hint_turns, extra: `hints injected ${m.hints_injected ?? 0}` },
    { arm: a.arm, mechanism: 'deep thinking', n, unit: 'turns', ran: m.investigate_ran, produced: m.investigate_injected },
    { arm: a.arm, mechanism: 'fan-out', n, unit: 'turns', ran: m.fanout_turns },
  ];
}

// ------------------------------------------------------------- headlines --

export type TileLine = { label: string; value: string; sub?: string };
export type Tile = { key: string; title: string; anchor: string; lines: TileLine[]; state: 'ready' | 'running' | 'empty' | 'error'; note?: string; published?: boolean };

export function summaryTiles(r: Results): Tile[] {
  const s = r.sections ?? {};
  const out: Tile[] = [];

  const lb = s.livebench;
  if (isError(lb)) out.push({ key: 'lb', title: 'LIVEBENCH', anchor: 'sec-livebench', lines: [], state: 'error', note: lb.error });
  else if (!lb || isEmpty(lb)) out.push({ key: 'lb', title: 'LIVEBENCH', anchor: 'sec-livebench', lines: [], state: 'empty', note: 'no run' });
  else {
    const run = currentRun(lb as LiveBenchSection);
    const scored = (run?.arms ?? []).filter((a) => a.overall?.score !== null && a.overall?.score !== undefined);
    const lines = scored.map((a) => ({
      label: a.arm,
      value: scoreCi(a.overall.score, a.overall.ci95),
      sub: `n ${lbN(a)} · ${(a.overall.categories_included ?? []).join(', ')}`,
    }));
    const answered = (run?.progress ?? []).reduce((x, p) => x + (p.answered ?? 0), 0);
    out.push({
      key: 'lb', title: 'LIVEBENCH', anchor: 'sec-livebench', lines,
      state: lines.length ? 'ready' : run ? 'running' : 'empty',
      note: run ? `${run.run_id}${lines.length ? '' : ` · ${answered} answered, not yet scored`}` : 'no current run',
    });
  }

  const swe = s.swebench;
  if (isError(swe)) out.push({ key: 'swe', title: 'SWE-BENCH VERIFIED', anchor: 'sec-swebench', lines: [], state: 'error', note: swe.error });
  else if (!swe || isEmpty(swe)) out.push({ key: 'swe', title: 'SWE-BENCH VERIFIED', anchor: 'sec-swebench', lines: [], state: 'empty', note: 'no run' });
  else {
    const w = swe as Extract<typeof swe, { arms: SweArm[] }>;
    out.push({
      key: 'swe', title: 'SWE-BENCH VERIFIED MINI', anchor: 'sec-swebench', state: w.arms?.length ? 'ready' : 'running',
      lines: (w.arms ?? []).map((a) => ({
        label: a.arm,
        value: `${a.resolved}/${a.scored} resolved`,
        sub: `Wilson ${pc0(a.lo)}–${pc0(a.hi)} · of ${w.planned ?? '?'} planned`,
      })),
      note: w.run_id,
    });
  }

  const dom = s.domain;
  if (isError(dom)) out.push({ key: 'dom', title: 'DOMAIN TASKS', anchor: 'sec-domain', lines: [], state: 'error', note: dom.error });
  else if (!dom || isEmpty(dom)) out.push({ key: 'dom', title: 'DOMAIN TASKS', anchor: 'sec-domain', lines: [], state: 'empty', note: 'no run' });
  else if (isState(dom, 'running')) {
    const d = dom as { group?: string; current?: { planned_pairs: number | null; scored_pairs: number } };
    out.push({
      key: 'dom', title: 'DOMAIN TASKS', anchor: 'sec-domain', lines: [], state: 'running',
      note: `${d.group ?? ''} · ${d.current?.scored_pairs ?? 0}/${d.current?.planned_pairs ?? '?'} pairs scored`,
    });
  } else {
    const d = dom as Extract<typeof dom, { arms: string[] }>;
    const pa = d.per_arm_headline ?? {};
    out.push({
      key: 'dom', title: 'DOMAIN TASKS', anchor: 'sec-domain', state: 'ready',
      lines: (d.arms ?? []).flatMap((a) => {
        const x = pa[a];
        return x
          ? [{ label: a, value: `${x.passed}/${x.scored} pass`, sub: `Wilson ${pc0(x.scored ? x.lo : null)}–${pc0(x.scored ? x.hi : null)}` }]
          : [];
      }),
      note: `${d.run_id ?? ''}${d.task_counts ? ` · ${d.task_counts.uncontaminated} uncontaminated tasks` : ''}`,
    });
  }

  const sp = s.speed;
  if (isError(sp)) out.push({ key: 'spd', title: 'DECODE SPEED', anchor: 'sec-speed', lines: [], state: 'error', note: sp.error });
  else if (sp && !isEmpty(sp)) {
    const rows = (sp as Extract<typeof sp, { mtp: unknown }>).mtp?.builds?.rows ?? [];
    const withTps = rows.filter((x) => x.mean_tps !== null);
    const best = withTps.reduce<(typeof rows)[number] | null>((b, x) => (b === null || (x.mean_tps ?? 0) > (b.mean_tps ?? 0) ? x : b), null);
    const base = withTps.find((x) => /production/i.test(x.build)) ?? null;
    const lines: TileLine[] = [];
    if (best) lines.push({ label: `run ${best.run}`, value: `${best.mean_tps} tok/s`, sub: `${best.build} · head ${best.head}` });
    if (base && base !== best) lines.push({ label: `run ${base.run}`, value: `${base.mean_tps} tok/s`, sub: base.build });
    out.push({
      key: 'spd', title: 'DECODE SPEED', anchor: 'sec-speed', lines, state: lines.length ? 'ready' : 'empty',
      note: (sp as Extract<typeof sp, { mtp: unknown }>).mtp?.section6 ?? undefined,
    });
  }

  const img = s.imagegen;
  if (isError(img)) out.push({ key: 'img', title: 'IMAGES', anchor: 'sec-images', lines: [], state: 'error', note: img.error });
  else if (!img || isEmpty(img)) out.push({ key: 'img', title: 'IMAGES', anchor: 'sec-images', lines: [], state: 'empty', note: 'no run' });
  else {
    const cs = ((img as Extract<typeof img, { configs: unknown }>).configs ?? []).filter((c) => c.images > 0 && c.median_wall_s !== null);
    const top = [...cs].sort((a, b) => b.images - a.images || (a.median_wall_s ?? 0) - (b.median_wall_s ?? 0))[0];
    out.push({
      key: 'img', title: 'IMAGES', anchor: 'sec-images', state: top ? 'ready' : 'empty',
      lines: top ? [{ label: top.config, value: `${f1(top.median_wall_s)} s / image`, sub: `median of ${top.images} · ${top.size}` }] : [],
    });
  }

  const card = s.model_card;
  if (card && !isEmpty(card) && !isError(card)) {
    const c = card as Extract<typeof card, { rows: unknown }>;
    const mean = c.rows.find((x) => /mean/i.test(x.bench));
    out.push({
      key: 'card', title: 'PUBLISHED MODEL CARD', anchor: 'sec-card', state: 'ready', published: true,
      lines: mean ? c.columns.map((col, i) => ({ label: col, value: String(mean.values[i] ?? '—'), sub: mean.bench })) : [],
      note: `published by ${c.publisher}, not measured here`,
    });
  }
  return out;
}
