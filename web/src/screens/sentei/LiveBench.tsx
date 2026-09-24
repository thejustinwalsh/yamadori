// LiveBench: the public 2024-11-25 questions, LiveBench's own judge, our arms
// against the reference models' published judgments on the same questions.
import { electricityCells, rangeText } from '../../api/power';
import type { LbArm, LbRef, LbRun, LiveBenchSection, Section } from '../../api/types';
import { ago, n } from '../../format';
import { Panel } from '../../ui/Panel';
import { Chip } from '../../ui/primitives';
import { StateView } from '../../ui/StateView';
import { Table } from '../../ui/Table';
import { currentRun, f1, isEmpty, isError, lbN, mechRowsLiveBench, scoreCi } from './model';
import { ArmsLegend, type Bar, Chips, Collapse, CompareBars, H3, Lines, MechTable, NotReady, PairedTable, tally, What } from './parts';

const pct100 = (x: number | null | undefined) => (x === null || x === undefined ? null : x / 100);

function OverallBars({ run }: { run: LbRun }) {
  const bars: Bar[] = run.arms
    .filter((a) => a.overall.score !== null)
    .map((a) => ({
      label: a.arm,
      value: pct100(a.overall.score),
      lo: pct100(a.overall.ci95?.[0]),
      hi: pct100(a.overall.ci95?.[1]),
      text: `${scoreCi(a.overall.score, a.overall.ci95)} · n ${lbN(a)}`,
      ours: true,
    }));
  return <CompareBars bars={bars} label="overall score per arm, bootstrap 95% interval" />;
}

function SameQuestions({ run }: { run: LbRun }) {
  const arms = run.arms.filter((a) => a.overall.score !== null && run.same_question_refs?.[a.arm]?.length);
  if (!arms.length) return null;
  const NAME: Record<string, string> = { instruction_following: 'instruction following', data_analysis: 'data analysis' };
  const charts: { key: string; label: string; bars: Bar[] }[] = [];
  for (const a of arms) {
    const refs = run.same_question_refs[a.arm] ?? [];
    const scopes: { key: string; name: string; ours: { score: number | null; ci95: [number, number] | null; n: number | null }; ref: (r: LbRef) => number | null }[] = [];
    if (refs.some((r) => r.overall !== null)) {
      scopes.push({ key: 'overall', name: 'overall', ours: { ...a.overall, n: lbN(a) }, ref: (r) => r.overall });
    }
    const cats = Object.keys(a.categories);
    // one category: its chart is the overall chart, drawn once
    for (const c of scopes.length && cats.length === 1 ? [] : cats) {
      const mine = a.categories[c];
      if (mine && refs.some((r) => typeof r.categories?.[c] === 'number')) {
        scopes.push({ key: c, name: NAME[c] ?? c, ours: mine, ref: (r) => r.categories?.[c] ?? null });
      }
    }
    for (const sc of scopes) {
      const rs = refs
        .map((r) => ({ r, v: sc.ref(r) }))
        .filter((x): x is { r: LbRef; v: number } => typeof x.v === 'number')
        .sort((x, y) => y.v - x.v);
      charts.push({
        key: `${a.arm}|${sc.key}`,
        label: `${a.arm} against reference models, ${sc.name}, same questions`,
        bars: [
          {
            label: `${a.arm} (ours)`,
            value: pct100(sc.ours.score),
            lo: pct100(sc.ours.ci95?.[0]),
            hi: pct100(sc.ours.ci95?.[1]),
            text: `${scoreCi(sc.ours.score, sc.ours.ci95)} · n ${sc.ours.n ?? '—'}`,
            ours: true,
            group: `${sc.name.toUpperCase()} · MEASURED HERE · n ${sc.ours.n ?? '—'} QUESTIONS`,
          },
          ...rs.map(({ r, v }) => ({
            label: r.model,
            value: pct100(v),
            text: f1(v),
            group: `${sc.name.toUpperCase()} · PUBLISHED PER-QUESTION JUDGMENTS · SAME QUESTIONS`,
          })),
        ],
      });
    }
  }
  const missing = arms.flatMap((a) => (run.same_question_refs[a.arm] ?? []).filter((r) => r.missing)).length;
  return (
    <>
      <H3>Same questions · ours against LiveBench's published judgments of other models</H3>
      {charts.map((c) => (
        <CompareBars key={c.key} bars={c.bars} label={c.label} />
      ))}
      {missing ? (
        <Lines items={['a category is charted only where LiveBench publishes per-question judgments for it on this release; overall needs every category']} />
      ) : null}
    </>
  );
}

function Categories({ run }: { run: LbRun }) {
  const arms = run.arms.filter((a) => Object.keys(a.categories).length);
  if (!arms.length) return null;
  type Row = { key: string; name: string; task: boolean; cells: Record<string, { score: number | null; ci95: [number, number] | null; n: number | null } | undefined> };
  const rows: Row[] = [];
  const cats = Array.from(new Set(arms.flatMap((a) => Object.keys(a.categories))));
  for (const c of cats) {
    rows.push({ key: c, name: c, task: false, cells: Object.fromEntries(arms.map((a) => [a.arm, a.categories[c]])) });
    const tasks = Array.from(new Set(arms.flatMap((a) => Object.keys(a.categories[c]?.tasks ?? {}))));
    for (const t of tasks) {
      rows.push({ key: `${c}/${t}`, name: `  ${t}`, task: true, cells: Object.fromEntries(arms.map((a) => [a.arm, a.categories[c]?.tasks?.[t]])) });
    }
  }
  return (
    <Table
      rows={rows}
      rowKey={(r) => r.key}
      caption="score per category and task, bootstrap 95% interval"
      columns={[
        { key: 'c', head: 'category / task', cell: (r) => (r.task ? <span style={{ paddingInlineStart: '1ch' }}>{r.name.trim()}</span> : <strong>{r.name}</strong>) },
        ...arms.map((a) => ({
          key: a.arm,
          head: `${a.arm} · score [CI] · n`,
          num: true,
          cell: (r: Row) => {
            const c = r.cells[a.arm];
            return c ? `${scoreCi(c.score, c.ci95)} · ${c.n ?? '—'}` : '—';
          },
        })),
      ]}
    />
  );
}

function StatusTable({ run }: { run: LbRun }) {
  if (!run.arms.some((a) => Object.keys(a.status).length)) return null;
  const secs = (a: LbArm) => Object.values(a.categories).map((c) => c.seconds?.median).filter((x): x is number => typeof x === 'number');
  const toks = (a: LbArm) => Object.values(a.categories).map((c) => c.output_tokens?.median).filter((x): x is number => typeof x === 'number');
  return (
    <Table
      rows={run.arms}
      rowKey={(a) => a.arm}
      caption="answer status and finish reason per arm"
      columns={[
        { key: 'a', head: 'arm', cell: (a) => a.arm },
        { key: 's', head: 'status', cell: (a) => tally(a.status) },
        { key: 'f', head: 'finish reason', cell: (a) => tally(a.finish_reason) },
        { key: 'sec', head: 'median s by category', num: true, cell: (a) => secs(a).map((x) => Math.round(x)).join(' / ') || '—' },
        { key: 'tok', head: 'median output tok', num: true, cell: (a) => toks(a).map((x) => n(Math.round(x))).join(' / ') || '—' },
      ]}
    />
  );
}

/** Electricity per question: an ESTIMATE from wall seconds x the measured generating draw (mcp/power.py). */
export function Electricity({ run }: { run: LbRun }) {
  const arms = run.arms.filter((a) => a.electricity);
  const b = run.electricity_basis;
  if (!arms.length || !b) return null;
  return (
    <>
      <H3>Electricity · estimate</H3>
      <Table
        rows={arms}
        rowKey={(a) => a.arm}
        caption="estimated electricity per question and per 100 questions, per arm"
        columns={[
          { key: 'a', head: 'arm', cell: (a) => a.arm },
          { key: 'n', head: 'n', num: true, cell: (a) => a.electricity!.n },
          { key: 's', head: 'wall s / question', num: true, cell: (a) => electricityCells(a.electricity)!.seconds },
          { key: 'wh', head: 'Wh / question', num: true, cell: (a) => electricityCells(a.electricity)!.perQuestionWh },
          { key: 'c', head: '¢ / question', num: true, cell: (a) => electricityCells(a.electricity)!.perQuestionCents },
          { key: 'k', head: 'kWh / 100', num: true, cell: (a) => electricityCells(a.electricity)!.per100Kwh },
          { key: 'd', head: '$ / 100', num: true, cell: (a) => electricityCells(a.electricity)!.per100Dollars },
        ]}
      />
      <Lines
        items={[
          `ESTIMATE, not metered: wall seconds per answered question × ${b.watts} W (${b.gpu_watts} W GPU${b.extra_watts ? ` + ${b.extra_watts} W configured extra` : ''}) · ${b.evidence}`,
          `GPU draw only · priced ${rangeText(b.cents_per_kwh, (x) => x.toFixed(3))}¢/kWh, the ${b.priced} · ${b.rate}`,
          b.overlap,
        ]}
      />
    </>
  );
}

function Paired({ run }: { run: LbRun }) {
  const rows = run.paired.flatMap((p) =>
    Object.entries(p.categories).map(([c, v]) => ({
      key: `${p.b}|${p.a}|${c}`,
      a: p.a,
      b: p.b,
      scope: c,
      n: v.n_pairs,
      bOnly: v.binary ? v.b_only_correct : null,
      aOnly: v.binary ? v.a_only_correct : null,
      p: v.mcnemar_p,
      diff: v.diff === null ? undefined : `${v.diff >= 0 ? '+' : ''}${f1(v.diff)} [${f1(v.ci95?.[0])}, ${f1(v.ci95?.[1])}]`,
    })),
  );
  const overall = run.paired.filter((p) => p.diff !== null);
  return (
    <>
      <PairedTable rows={rows} caption="each arm against bonsai on the questions both scored" />
      {overall.length ? (
        <Chips>
          {overall.map((p) => (
            <Chip key={`${p.b}|${p.a}`} tone="muted">
              {p.b} − {p.a} OVERALL {p.diff! >= 0 ? '+' : ''}
              {f1(p.diff)} [{f1(p.ci95?.[0])}, {f1(p.ci95?.[1])}] · {p.n_pairs} PAIRS
            </Chip>
          ))}
        </Chips>
      ) : null}
      {rows.some((r) => r.bOnly === null) ? (
        <Lines items={['non-binary scores (partial credit): no discordant count or McNemar test, only the bootstrap difference']} />
      ) : null}
    </>
  );
}

function Progress({ run }: { run: LbRun }) {
  if (!run.progress.length) return null;
  return (
    <Table
      rows={run.progress}
      rowKey={(p) => `${p.arm}|${p.category}`}
      caption="answering progress from the run's progress logs"
      columns={[
        { key: 'a', head: 'arm', cell: (p) => p.arm },
        { key: 'c', head: 'category', cell: (p) => p.category },
        { key: 'n', head: 'answered', num: true, cell: (p) => `${p.answered}/${p.of ?? '—'}` },
        { key: 'e', head: 'transport errors', num: true, cell: (p) => p.err_rows },
        { key: 't', head: 'last chunk', num: true, cell: (p) => p.at ?? '—' },
        { key: 'g', head: 'log written', num: true, cell: (p) => `${ago(p.log_age_s)} ago` },
      ]}
    />
  );
}

function Published({ run }: { run: LbRun }) {
  const cats = run.arms.flatMap((a) => a.overall.categories_included);
  const pub = run.published_2024_11_25 ?? {};
  const rows = Object.entries(pub)
    .map(([m, v]) => ({ m, v }))
    .sort((x, y) => (y.v.global ?? 0) - (x.v.global ?? 0));
  if (!rows.length) return null;
  const NAME: Record<string, string> = { coding: 'Coding', instruction_following: 'IF', reasoning: 'Reasoning', math: 'Mathematics', data_analysis: 'Data Analysis', language: 'Language' };
  const colsUsed = Array.from(new Set(cats)).map((c) => NAME[c] ?? c);
  return (
    <Collapse summary={`Published 2024-11-25 leaderboard · full release, every question · ${rows.length} models`}>
      <Table
        rows={rows}
        rowKey={(r) => r.m}
        tall
        caption="published LiveBench 2024-11-25 table, all questions"
        columns={[
          { key: 'm', head: 'model (published)', cell: (r) => r.m },
          { key: 'g', head: 'global, 6 categories', num: true, cell: (r) => f1(r.v.global) },
          ...colsUsed.map((c) => ({ key: c, head: c, num: true, cell: (r: (typeof rows)[number]) => f1(r.v.categories?.[c]) })),
        ]}
      />
    </Collapse>
  );
}

function RunView({ run }: { run: LbRun }) {
  const cond = run.condition ?? {};
  const health = run.arms.flatMap((a) => mechRowsLiveBench(a.arm, a.mechanism_health));
  const unrecorded = run.arms.filter((a) => a.mechanism_health && a.mechanism_health.answers > 0 && !a.mechanism_health.recorded);
  return (
    <>
      <Chips>
        <Chip tone={run.frozen ? 'muted' : run.state === 'ready' ? 'moss' : 'cyan'}>
          {run.run_id} · {run.frozen ? 'FROZEN RECORD' : run.state === 'ready' ? 'CURRENT' : 'RUNNING · NOT YET SCORED'}
        </Chip>
        {cond.min_p !== undefined && <Chip tone="muted">MIN_P {String(cond.min_p)}</Chip>}
        {run.release && <Chip tone="muted">RELEASE {Array.isArray(run.release) ? run.release.join(', ') : run.release}</Chip>}
        {cond.requests_in_flight !== undefined && <Chip tone="muted">{String(cond.requests_in_flight)} IN FLIGHT</Chip>}
        {cond.shared_with ? <Chip tone="rose">GPU SHARED</Chip> : null}
        {run.summary_age_s !== null && <Chip tone="muted">SUMMARY {ago(run.summary_age_s)} OLD</Chip>}
      </Chips>
      {run.frozen ? <Lines items={[run.frozen]} /> : null}
      <Lines
        items={[
          cond.shared_with ? `GPU shared with: ${String(cond.shared_with)}` : null,
          cond.server_sampling ? `server sampling: ${String(cond.server_sampling)}` : null,
        ].filter((x): x is string => !!x)}
      />
      <ArmsLegend rows={run.legend} />
      {run.state === 'ready' ? (
        <>
          <OverallBars run={run} />
          <Categories run={run} />
          <SameQuestions run={run} />
          <Paired run={run} />
          <StatusTable run={run} />
          <Electricity run={run} />
          <H3>Mechanism health</H3>
          {health.length ? (
            <MechTable
              rows={health}
              caption="mechanism health per arm (bench/livebench/mechanisms.py)"
              stackErrors={Object.fromEntries(run.arms.map((a) => [a.arm, a.mechanism_health?.stack_errors ?? 0]))}
            />
          ) : null}
          {unrecorded.length ? (
            <Lines items={unrecorded.map((a) => `${a.arm}: ${a.mechanism_health.answers} answers scored before per-answer mechanism records existed · 0 recorded`)} />
          ) : null}
          <Published run={run} />
        </>
      ) : (
        <StateView kind="loading" title={`${run.run_id} · answering · no summary.json yet`} />
      )}
      <Progress run={run} />
      {run.method?.bootstrap_B ? (
        <Lines items={[`interval: ${run.method.ci}, B=${run.method.bootstrap_B} · aggregation: ${run.method.aggregation}`]} />
      ) : null}
    </>
  );
}

export function LiveBench({ sec }: { sec: Section<LiveBenchSection> | undefined }) {
  const ready = sec && !isEmpty(sec) && !isError(sec) ? (sec as LiveBenchSection) : null;
  const cur = ready ? currentRun(ready) : null;
  const records = ready ? ready.runs.filter((r) => r !== cur) : [];
  return (
    <Panel id="sec-livebench" kanji="問" title="LIVEBENCH" tag="SCORE 0–100" tagTone="cyan" fill>
      <What>
        General capability on LiveBench's public 2024-11-25 questions (coding, instruction following, …), scored by LiveBench's own
        judge. Reference models are LiveBench's published judgments on exactly the questions we answered.
      </What>
      {!ready ? (
        <NotReady sec={sec as never} what="LiveBench" />
      ) : (
        <>
          {cur ? <RunView run={cur} /> : <StateView kind="empty" title="no current (unfrozen) run" />}
          {records.map((r) => (
            <Collapse key={r.run_id} summary={`Record · ${r.run_id}${r.condition?.min_p !== undefined ? ` · min_p ${r.condition.min_p}` : ''}${r.frozen ? ' · frozen' : ''}`}>
              <RunView run={r} />
            </Collapse>
          ))}
          {cur?.current_leaderboard_base ? (
            <Lines
              items={Object.entries(cur.current_leaderboard_base.models).map(
                ([m, v]) => `${m}: global ${f1(v.global)} on the ${cur.current_leaderboard_base!.date} leaderboard · different, unpublished questions · not comparable`,
              )}
            />
          ) : null}
        </>
      )}
    </Panel>
  );
}

