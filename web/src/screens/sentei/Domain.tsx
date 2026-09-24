// The domain suite (bench/domain): this stack's own languages, graded by
// compile and hidden tests, each arm paired against A0 on the same tasks.
import * as stylex from '@stylexjs/stylex';
import type { DomainArmStats, DomainBlock, DomainExtras, DomainGroup, DomainRunning, LcbArm, LcbSection, Section } from '../../api/types';
import { ago, n, pct, pValue } from '../../format';
import { Panel } from '../../ui/Panel';
import { Chip, SplitBar } from '../../ui/primitives';
import { RateBar } from '../../ui/RateBar';
import { Table } from '../../ui/Table';
import { f1, isEmpty, isError, isState, kOfN, mechRowsDomain, mechRowsDomainHealth, pc0 } from './model';
import { ArmsLegend, type Bar, Chips, Collapse, CompareBars, H3, Lines, MechTable, NotReady, type PairRow, PairedTable, tally, What } from './parts';

type Dom = LcbSection & DomainExtras;

const STAGE_TONE = { pass: 'moss', test: 'cyan', compile: 'crimson', extract: 'muted' } as const;

function Stages({ arm, st }: { arm: string; st: Record<string, number> | undefined }) {
  if (!st) return <>—</>;
  const parts = (['pass', 'test', 'compile', 'extract'] as const).map((k) => ({ value: st[k] ?? 0, tone: STAGE_TONE[k] }));
  return (
    <span style={{ display: "block", minWidth: 96 }} title={`pass ${st.pass ?? 0} · failed at test ${st.test ?? 0} · compile ${st.compile ?? 0} · extract ${st.extract ?? 0}`}>
      <SplitBar parts={parts} label={`${arm}: pass ${st.pass ?? 0}, failed at test ${st.test ?? 0}, compile ${st.compile ?? 0}, extract ${st.extract ?? 0}`} />
    </span>
  );
}

function pairsOf(block: DomainBlock | undefined, scope: string): PairRow[] {
  if (!block?.tasks) return [];
  return (block.comparisons ?? []).map((c) => ({
    key: `${scope}|${c.a}|${c.b}`,
    a: c.a,
    b: c.b,
    scope,
    n: c.n_paired,
    aPass: c.a_pass,
    bPass: c.b_pass,
    bOnly: c.b_only,
    aOnly: c.a_only,
    p: c.p_bonferroni ?? c.p,
    pLabel: c.family_size && c.family_size > 1 ? `(Bonferroni ×${c.family_size})` : undefined,
    floor: c.family ? (block?.families?.[c.family]?.min_discordant_for_sig ?? null) : null,
  }));
}

function HeadlineArms({ d }: { d: Dom }) {
  const pa = d.per_arm_headline ?? {};
  const rows = (d.arms ?? []).flatMap((a) => (pa[a] ? [{ arm: a, ...pa[a] }] : [])) as (DomainArmStats & { arm: string })[];
  if (!rows.length) return null;
  const conc = d.timing && !d.timing.clean;
  const bars: Bar[] = rows.map((r) => ({
    label: r.arm,
    value: r.scored ? r.rate : null,
    lo: r.scored ? r.lo : null,
    hi: r.scored ? r.hi : null,
    text: kOfN(r.passed, r.scored, r.scored ? r.lo : null, r.scored ? r.hi : null),
    ours: true,
  }));
  return (
    <>
      <H3>Pass rate per arm · uncontaminated tasks · each arm on every task it finished</H3>
      <CompareBars bars={bars} label="pass rate per arm with Wilson 95% interval" />
      <Table
        rows={rows}
        rowKey={(r) => r.arm}
        caption="per arm: where failures died, errors excluded, cost"
        columns={[
          { key: 'a', head: 'arm', cell: (r) => r.arm },
          { key: 'k', head: 'pass / scored', num: true, cell: (r) => `${r.passed}/${r.scored}` },
          { key: 'st', head: 'pass · test · compile · extract', cell: (r) => <Stages arm={r.arm} st={r.stages as Record<string, number> | undefined} /> },
          { key: 'fs', head: 'failed at', cell: (r) => tally(r.fail_stages) },
          { key: 'se', head: 'stack errors (not scored)', num: true, cell: (r) => (r.stack_error_rows ? `${r.stack_error_rows} · ${tally(r.stack_error_kinds)}` : 0) },
          { key: 's', head: `median s${conc ? ' · concurrent' : ''}`, num: true, cell: (r) => f1(r.median_seconds) },
          { key: 'ti', head: 'median prompt / completion tok', num: true, cell: (r) => `${n(r.median_prompt_tokens)} / ${n(r.median_completion_tokens)}` },
          { key: 'fc', head: 'answer compiles', num: true, cell: (r) => (r.final_compiles === undefined ? '—' : `${r.final_compiles}/${r.scored}`) },
          { key: 'th', head: 'tool rounds · deep-thinking searches (mean)', num: true, cell: (r) => `${f1(r.tool_hops_mean)} · ${f1(r.investigate_hops_mean)}` },
        ]}
      />
    </>
  );
}

function ByDomain({ d }: { d: Dom }) {
  const rows = Object.entries(d.by_domain ?? {}).map(([name, v]) => ({ name, ...v }));
  if (!rows.length) return null;
  const arms = d.arms ?? [];
  return (
    <>
      <H3>Per domain · Wilson 95%</H3>
      <Table
        rows={rows}
        rowKey={(x) => x.name}
        caption="pass rate per domain and arm"
        columns={[
          {
            key: 'd',
            head: 'domain',
            cell: (x) => (
              <span>
                {x.name} {x.contaminated ? <Chip tone="rose">CONTAMINATED</Chip> : null}
              </span>
            ),
          },
          { key: 't', head: 'tasks', num: true, cell: (x) => x.tasks },
          ...arms.map((a) => ({
            key: a,
            head: a,
            cell: (x: (typeof rows)[number]) => {
              const c = x.by_arm?.[a];
              if (!c || !c.n) return c?.stack_errors ? `${c.stack_errors} stack err` : '—';
              return (
                <div style={{ display: 'grid', gap: 2, minWidth: 110 }}>
                  <span title={`failed at test ${c.stages?.test ?? 0} · compile ${c.stages?.compile ?? 0} · extract ${c.stages?.extract ?? 0}`}>
                    {c.k}/{c.n} · {pc0(c.rate)} [{pc0(c.lo)}–{pc0(c.hi)}]
                  </span>
                  <RateBar rate={c.rate} lo={c.lo} hi={c.hi} tone={x.contaminated ? 'rose' : 'moss'} label={`${a} ${x.name} ${pct(c.rate, 0)}`} />
                </div>
              );
            },
          })),
        ]}
      />
    </>
  );
}

function Twins({ d }: { d: Dom }) {
  if (!d.twins?.length) return null;
  return (
    <Table
      rows={d.twins}
      rowKey={(t) => `${t.one_shot}|${t.s}`}
      caption="self-check arm against its one-shot twin"
      columns={[
        { key: 'ab', head: 'self-check vs one-shot', cell: (t) => `${t.s} vs ${t.one_shot}` },
        { key: 'n', head: 'paired', num: true, cell: (t) => t.n_paired },
        { key: 'pp', head: 'pass one-shot / S', num: true, cell: (t) => `${t.one_shot_pass} / ${t.s_pass}` },
        { key: 'so', head: 'S only', num: true, cell: (t) => t.s_only },
        { key: 'oo', head: 'one-shot only', num: true, cell: (t) => t.one_shot_only },
        { key: 'p', head: 'McNemar p', num: true, cell: (t) => pValue(t.p_bonferroni ?? t.p) },
        { key: 'fx', head: 'compile fail → pass', num: true, cell: (t) => t.fixed_by_checking },
        { key: 'ct', head: 'compile → test', num: true, cell: (t) => t.compile_to_test },
        { key: 'tp', head: 'test → pass', num: true, cell: (t) => t.test_to_pass },
        { key: 'cf', head: 'compile fails one-shot / S', num: true, cell: (t) => `${t.one_shot_compile_fail} / ${t.s_compile_fail}` },
      ]}
    />
  );
}

function Style({ d }: { d: Dom }) {
  const rows = d.arm_table.filter((a) => (a.lint_n ?? 0) > 0);
  if (!rows.length) return null;
  return (
    <>
      <H3>Lint and modern-idiom score · separate from pass/fail</H3>
      <Table
        rows={rows}
        rowKey={(a: LcbArm) => a.arm}
        caption="style score per arm (grade_style.py)"
        columns={[
          { key: 'arm', head: 'arm', cell: (a) => a.arm },
          { key: 'ln', head: 'linted', num: true, cell: (a) => a.lint_n ?? 0 },
          { key: 'lc', head: '0 lint errors', num: true, cell: (a) => `${a.lint_clean ?? 0}/${a.lint_n ?? 0}` },
          { key: 'le', head: 'errors mean', num: true, cell: (a) => f1(a.lint_errors_mean) },
          { key: 'lw', head: 'warnings mean', num: true, cell: (a) => f1(a.lint_warnings_mean) },
          { key: 'mf', head: 'outdated-idiom flags', cell: (a) => tally(a.modern_flags) },
          { key: 'mc', head: 'modern-idiom credits', cell: (a) => tally(a.modern_credits) },
        ]}
      />
    </>
  );
}

const s = stylex.create({ wrap: { display: 'contents' } });

const cond = (c: Record<string, unknown> | undefined) =>
  Object.entries(c ?? {})
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `${k} ${String(v)}`)
    .join(' · ');

/** Every run of the suite: what it planned, holds, and excluded as stale, with the reason. */
function Groups({ groups, current }: { groups: DomainGroup[] | undefined; current?: string }) {
  if (!groups?.length) return null;
  return (
    <Collapse summary={`All domain runs · ${groups.length} groups · stale rows are excluded, with the reason`} open={!groups.some((g) => g.group === current && g.rows > 0)}>
      <Table
        rows={groups}
        rowKey={(g) => g.group}
        caption="every bench/domain group"
        columns={[
          { key: 'g', head: 'group', cell: (g) => (g.group === current ? <strong>{g.group} · shown</strong> : g.group) },
          { key: 'a', head: 'arms', cell: (g) => g.arms.join(' · ') },
          { key: 'c', head: 'conditions', cell: (g) => cond(g.conditions) || '—' },
          { key: 'p', head: 'scored / planned pairs', num: true, cell: (g) => `${g.scored_pairs}/${g.planned_pairs ?? '—'}` },
          { key: 'r', head: 'rows', num: true, cell: (g) => g.rows },
          {
            key: 's',
            head: 'excluded (stale)',
            cell: (g) =>
              g.stale.length ? (
                <span>
                  {g.stale.map((x) => `${x.rows} rows · ${x.reason}`).join('; ')}
                </span>
              ) : (
                '0'
              ),
          },
          { key: 'w', head: 'written', num: true, cell: (g) => `${ago(g.age_s)} ago` },
        ]}
      />
    </Collapse>
  );
}

function Running({ r }: { r: DomainRunning }) {
  const c = r.current;
  return (
    <>
      <Chips>
        <Chip tone="cyan">{r.group ?? r.run_id} · RUNNING · NOTHING SCORED YET</Chip>
        {c && <Chip tone="muted">{c.scored_pairs}/{c.planned_pairs ?? '?'} PAIRS SCORED</Chip>}
        {Object.entries(c?.conditions ?? {}).map(([k, v]) => (
          <Chip key={k} tone="muted">
            {k.toUpperCase().replace('_', ' ')} {String(v)}
          </Chip>
        ))}
        {(c?.dirs.length ?? 0) > 1 &&
          c!.dirs.map((x) => (
            <Chip key={x} tone="muted">
              RUNNER {x}
            </Chip>
          ))}
      </Chips>
      <Lines items={(c?.stale ?? []).map((x) => `excluded, stale_code: ${x.rows} rows (${tally(x.arms)}) · ${x.reason}`)} warn />
      <ArmsLegend rows={r.legend} />
      <Groups groups={r.groups} current={r.group} />
    </>
  );
}

export function Domain({ sec }: { sec: Section<Dom> | DomainRunning | undefined }) {
  const running = isState(sec, 'running') ? (sec as DomainRunning) : null;
  const d = sec && !running && !isEmpty(sec) && !isError(sec) ? (sec as Dom) : null;
  const conds = d?.conditions ?? {};
  const mech = d
    ? (d.arms ?? []).flatMap((a) => {
        const h = mechRowsDomainHealth(a, d.per_arm_headline?.[a]?.mechanism_health);
        return h.length ? h : mechRowsDomain(a, d.triggers?.[a]);
      })
    : [];
  const headPairs = d ? pairsOf(d.headline, 'uncontaminated') : [];
  const dirtyPairs = d ? pairsOf(d.contaminated_block, 'contaminated only') : [];
  const domPairs: PairRow[] = d
    ? Object.entries(d.by_domain ?? {}).flatMap(([dom, v]) =>
        (v.vs_A0 ?? []).map((c) => ({
          key: `${dom}|${c.a}|${c.b}`, a: c.a, b: c.b, scope: `${dom}${v.contaminated ? ' (contaminated)' : ''}`,
          n: c.n_paired, aPass: c.a_pass, bPass: c.b_pass, bOnly: c.b_only, aOnly: c.a_only, p: c.p, pLabel: '(uncorrected)',
        })),
      )
    : [];
  return (
    <Panel id="sec-domain" kanji="盆" title="DOMAIN TASKS" tag="PASS@1" tagTone="cyan" fill>
      <What>
        This stack's own work: TypeScript, Rust/WASM across C ABIs, React and three.js TSL tasks, each graded by compile and hidden
        tests. Every arm runs the same tasks, so each is paired against A0 (the bare model) task by task.
      </What>
      {running ? (
        <Running r={running} />
      ) : !d ? (
        <NotReady sec={sec as never} what="domain suite" />
      ) : (
        <div {...stylex.props(s.wrap)}>
          <Chips>
            <Chip tone="muted">{d.run_id ?? d.file}</Chip>
            {(d.merged_run_dirs?.length ?? 0) > 1 &&
              d.merged_run_dirs!.map((x) => (
                <Chip key={x} tone="muted">
                  MERGED {x}
                </Chip>
              ))}
            {['effort', 'temperature', 'suite', 'body_tier'].filter((k) => conds[k] !== undefined && conds[k] !== null).map((k) => (
              <Chip key={k} tone="muted">
                {k.toUpperCase().replace('_', ' ')} {String(conds[k])}
              </Chip>
            ))}
            {d.task_counts && (
              <Chip tone="muted">
                {d.task_counts.uncontaminated} TASKS UNCONTAMINATED · {d.task_counts.contaminated} CONTAMINATED
              </Chip>
            )}
            {d.timing && !d.timing.clean && <Chip tone="rose">TIMING CONCURRENT · {d.timing.concurrent_with.join(' · ')}</Chip>}
            {(d.stale_rows ?? 0) > 0 && <Chip tone="rose">{d.stale_rows} STALE ROWS EXCLUDED</Chip>}
          </Chips>
          <Lines
            items={(d.stale ?? []).map((x) => `excluded, stale_code: ${x.rows} row${x.rows === 1 ? '' : 's'} (${tally(x.arms)}) · ${x.reason}`)}
            warn
          />
          <Lines items={(d.caveats ?? []).filter((c) => !c.includes('stale_code'))} warn />
          <ArmsLegend rows={d.legend} />
          <HeadlineArms d={d} />
          <H3>Paired against A0 · the discordant tasks are what the exact test sees</H3>
          <PairedTable rows={[...headPairs, ...dirtyPairs]} caption="each arm against A0 on the tasks both scored" />
          <ByDomain d={d} />
          {domPairs.length ? (
            <Collapse summary={`Paired against A0 per domain · ${domPairs.length} comparisons, uncorrected`}>
              <PairedTable rows={domPairs} caption="per domain, each arm against A0" />
            </Collapse>
          ) : null}
          <Twins d={d} />
          <Style d={d} />
          {mech.length ? (
            <>
              <H3>Mechanism health · read off each answer's x_yamadori</H3>
              <MechTable rows={mech} caption="per arm: mechanisms allowed, chosen, ran, produced" />
            </>
          ) : null}
          <Groups groups={d.groups} current={d.group} />
        </div>
      )}
    </Panel>
  );
}
