// 技 One skill: its SKILL.md (rendered and raw), each item with the source
// quote it rests on, where it sits on the taxonomy, its size against the
// caps, its activation tests and their last results, its stored behaviour
// checks, every version with every stage's output, its jobs, when it was
// selected, and what an operator can do to it. Everything from
// GET /dash/api/skills/<id>; the caps from GET /dash/api/skills.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useMemo, useState } from 'react';
import {
  canRerun,
  facetsOf,
  itemsOf,
  licenceOf,
  SKILL_POST,
  SKILLS_PATH,
  skillPath,
  splitRemedy,
  splitSkillMd,
  stageOutput,
  statusTone,
  versionText,
  type ActivationRun,
  type CaseResult,
  type Limits,
  type SkillDetail,
  type SkillsOverview,
  type SkillTests,
  type SkillVersion,
  type TestCase,
} from '../api/skills';
import { usePoll } from '../api/usePoll';
import { ago, n } from '../format';
import { href } from '../router';
import { space } from '../tokens/tokens.stylex';
import { NavLink } from '../ui/NavLink';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Row, Stat } from '../ui/primitives';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';
import { Btn, Chips, Field, Markdown, Note, Raw, s as p, send, SizeBar, StageRail, SubNav, type Msg } from './skills/parts';

const s = stylex.create({
  page: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'minmax(0, 3fr) minmax(0, 2fr)' },
    alignItems: 'start',
  },
  col: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  axis: { display: 'grid', gridTemplateColumns: 'minmax(6rem, auto) minmax(0, 1fr)', gap: space.spaceXs, alignItems: 'start' },
  item: { display: 'flex', flexDirection: 'column', gap: '4px', padding: space.spaceXs, minWidth: 0, textTransform: 'none' },
  fields: { display: 'flex', flexWrap: 'wrap', gap: space.spaceSm, alignItems: 'flex-end' },
});

// ---------------------------------------------------------------------------
function Header({ x, l, stale }: { x: SkillDetail; l: Limits | null; stale: boolean }) {
  const why = splitRemedy(x.reason);
  const lic = x.provenance?.licence;
  return (
    <Panel kanji="技" title={x.title || x.name} tag={x.status.toUpperCase()} tagTone={statusTone(x.status)} flag={x.id} stale={stale} edge={statusTone(x.status)} sub={`${x.folder ?? x.name} · ${x.source_kind}`}>
      {x.description ? <p {...stylex.props(text.bodySm, p.why)}>{x.description}</p> : null}
      {why.situation ? (
        <div {...stylex.props(layout.stack)}>
          <p {...stylex.props(text.bodySm, x.status === 'armed' ? p.why : p.err)}>{why.situation}</p>
          {why.remedy ? <p {...stylex.props(text.bodySm, p.why)}>REMEDY {why.remedy}</p> : null}
        </div>
      ) : null}
      <div {...stylex.props(layout.grid4)}>
        <Stat label="served" value={versionText(x)} />
        <Stat label="licence" value={licenceOf(x)} sub={lic?.where ?? undefined} />
        <Stat
          label="activation"
          value={x.activation?.score === null || x.activation?.score === undefined ? '—' : `${Math.round(x.activation.score * 100)}%`}
          sub={x.activation ? `${x.activation.passed ? 'passed' : 'FAILED'} · ${x.activation.n ?? 0} case(s)` : 'no result'}
          tone={x.activation && !x.activation.passed ? 'crimson' : undefined}
        />
        <Stat label="updated" value={`${ago(Date.now() / 1000 - x.updated)} ago`} sub={x.watch_seconds ? `re-fetch every ${n(x.watch_seconds / 3600)} h` : undefined} />
      </div>
      {l ? <SizeBar size={x.size} limits={l} /> : null}
      {lic?.quote ? <p {...stylex.props(text.bodySm, p.quote)}>“{lic.quote}”</p> : null}
      {x.provenance?.source || x.source_url ? (
        <Label>
          SOURCE · {x.provenance?.source ?? ''} {x.source_url ?? ''}
        </Label>
      ) : null}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function Taxonomy({ x, o }: { x: SkillDetail; o: SkillsOverview | null }) {
  const names = useMemo(() => Object.fromEntries((o ? facetsOf(o) : []).map((f) => [f.key, f.names])), [o]);
  const c = x.category;
  const g = x.gates;
  const rows: [string, string[] | undefined, Record<string, string> | undefined][] = [
    ['artifact', c?.artifact, names.artifact],
    ['language', c?.language, names.language],
    ['framework', c?.framework, names.framework],
    ['phase', c?.phase, names.phase],
    ['domain', c?.domain, undefined],
    ['situations', g?.situations, undefined],
    ['tools: any', g?.tools_any, undefined],
    ['tools: all', g?.tools_all, undefined],
    ['tools: none', g?.tools_none, undefined],
    ['all of', g?.all_of, undefined],
    ['topics', g?.topics, undefined],
    ['tags', x.tags, undefined],
  ];
  return (
    <Panel kanji="類" title="TAXONOMY · GATES" tag={x.applies_when ? 'APPLIES WHEN' : undefined} tagTone="muted">
      {x.applies_when ? <p {...stylex.props(text.bodySm, p.why)}>{x.applies_when}</p> : null}
      <div {...stylex.props(s.axis)}>
        {rows.map(([k, v, nm]) => (
          <Row key={k}>
            <Label>{k}</Label>
            <Chips values={v} names={nm} tone={k.startsWith('tools: none') ? 'rose' : 'muted'} />
          </Row>
        ))}
      </div>
      {x.triggers.length || x.learned_triggers.length ? (
        <div {...stylex.props(layout.stack)}>
          <Label>TRIGGERS</Label>
          {x.triggers.map((t, i) => (
            <Row key={`t${i}`}>
              <span {...stylex.props(p.why)}>{t}</span>
            </Row>
          ))}
          {x.learned_triggers.map((t, i) => (
            <Row key={`l${i}`}>
              <span {...stylex.props(p.why)}>{t}</span>
              <Chip tone="cyan">learned</Chip>
            </Row>
          ))}
        </div>
      ) : null}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function SkillMd({ x, busy, onSave }: { x: SkillDetail; busy: boolean; onSave: (text: string) => void }) {
  const [mode, setMode] = useState<'rendered' | 'raw' | 'edit'>('rendered');
  const [draft, setDraft] = useState('');
  const md = x.skill_md ?? '';
  const parts = splitSkillMd(md);
  return (
    <Panel
      kanji="書"
      title="SKILL.md"
      tag={x.text_version ? `v${x.text_version}` : 'NO TEXT'}
      tagTone={x.text_version ? 'cyan' : 'muted'}
      sub={mode === 'edit' ? 'a whole SKILL.md or just the body; the skill is disarmed until the edit re-passes the screen, its tests and validation' : 'what reaches the model is the title and the items, never the frontmatter'}
    >
      <div {...stylex.props(layout.rowWrap)}>
        <Btn disabled={mode === 'rendered'} onClick={() => setMode('rendered')}>
          RENDERED
        </Btn>
        <Btn disabled={mode === 'raw'} onClick={() => setMode('raw')}>
          RAW
        </Btn>
        <Btn
          disabled={mode === 'edit' || busy}
          onClick={() => {
            setDraft(md);
            setMode('edit');
          }}
        >
          EDIT
        </Btn>
      </div>
      {!md && mode !== 'edit' ? <StateView kind="inert" title="no validated text yet" /> : null}
      {md && mode === 'rendered' ? (
        <>
          {parts.frontmatter ? <Raw label="frontmatter" value={parts.frontmatter} /> : null}
          <Markdown md={parts.body} />
        </>
      ) : null}
      {md && mode === 'raw' ? <pre {...stylex.props(text.bodySm, p.pre)}>{md}</pre> : null}
      {mode === 'edit' ? (
        <>
          <textarea aria-label="SKILL.md" value={draft} onChange={(e) => setDraft(e.target.value)} {...stylex.props(text.bodySm, p.input, p.area, p.areaTall)} />
          <div {...stylex.props(layout.rowWrap)}>
            <Btn disabled={busy || !draft.trim() || draft === md} onClick={() => onSave(draft)}>
              SAVE · RE-SCREEN · RE-TEST
            </Btn>
            <Btn onClick={() => setMode('rendered')}>CANCEL</Btn>
          </div>
        </>
      ) : null}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function Items({ x, l }: { x: SkillDetail; l: Limits | null }) {
  const items = itemsOf(x);
  const shown = x.versions.find((v) => v.version === (x.text_version ?? x.served_version)) ?? x.versions[0];
  const dropped = shown?.validate?.dropped ?? [];
  const counts = shown?.validate?.counts;
  return (
    <Panel kanji="項" title="ITEMS · SOURCE QUOTES" tag={l ? `${items.length}/${l.max_items}` : `${items.length}`} tagTone="cyan" sub={counts ? `proposed ${counts.proposed ?? '—'} · kept ${counts.kept ?? '—'} · dropped ${counts.dropped ?? 0}` : undefined}>
      {items.length ? (
        items.map((it, i) => {
          const neg = /^(DO NOT|DON'T|NEVER)/i.test(it.form);
          const long = l && it.text.length + (it.situation?.length ?? 0) > l.max_item_chars;
          return (
            <Row key={i} bad={!!long}>
              <div {...stylex.props(s.item)}>
                <span {...stylex.props(text.bodySm)}>
                  <span {...stylex.props(neg ? p.formNot : p.form)}>
                    {it.form}
                    {it.situation ? ` ${it.situation}` : ''}:
                  </span>{' '}
                  {it.text}
                </span>
                {it.quote ? <p {...stylex.props(text.bodySm, p.quote)}>“{it.quote}”</p> : <span {...stylex.props(text.labelXs, text.outline)}>no source quote{it.provenance ? ` · ${it.provenance}` : ''}</span>}
                {it.ref ? <span {...stylex.props(text.labelXs, text.outline)}>ref {it.ref}</span> : null}
              </div>
            </Row>
          );
        })
      ) : (
        <StateView kind="inert" title="no validated items on this version" />
      )}
      {dropped.length ? (
        <details>
          <summary {...stylex.props(text.labelXs, p.summary)}>{dropped.length} item(s) dropped at validate</summary>
          {dropped.map((d, i) => (
            <Row key={i} bad>
              <span {...stylex.props(p.why)}>{d.item ?? '—'}</span>
              <span {...stylex.props(p.err)}>{d.why}</span>
            </Row>
          ))}
        </details>
      ) : null}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function caseText(c: TestCase): string {
  const bits = [c.text];
  if (c.files?.length) bits.push(`files: ${c.files.join(', ')}`);
  if (c.tool_output) bits.push(`tool output: ${c.tool_output}`);
  if (c.route_class) bits.push(`route: ${c.route_class}`);
  return bits.join(' · ');
}

/** A stored case's last result: the result list is should cases then should_not, each in order. */
function resultFor(run: ActivationRun | undefined, kind: 'should' | 'should_not', i: number): CaseResult | undefined {
  return run?.cases?.filter((c) => c.kind === kind)[i];
}

function CaseRows({ cases, kind, run }: { cases: TestCase[]; kind: 'should' | 'should_not'; run: ActivationRun | undefined }) {
  if (!cases.length) return <StateView kind="empty" title={`no ${kind === 'should' ? 'should' : 'near-miss'} cases`} />;
  return (
    <Table
      rows={cases.map((c, i) => ({ c, r: resultFor(run, kind, i), i }))}
      rowKey={(x) => String(x.i)}
      bad={(x) => x.r?.ok === false}
      columns={[
        { key: 'c', head: kind === 'should' ? 'should select it' : 'near miss: must not', cell: (x) => <span {...stylex.props(p.why)}>{caseText(x.c)}</span> },
        {
          key: 'r',
          head: 'last result',
          cell: (x) =>
            x.r ? (
              <span {...stylex.props(layout.stack)}>
                <Chip tone={x.r.ok ? 'moss' : 'crimson'}>{`${x.r.ok ? 'ok' : 'FAIL'} · ${x.r.verdict ?? '—'}`}</Chip>
                <span {...stylex.props(text.labelXs, text.outline)}>
                  {x.r.strength ?? ''} {x.r.gate ? `gate: ${x.r.gate}` : ''}
                  {x.r.rank !== undefined && x.r.rank !== null ? ` · rank ${x.r.rank}` : ''}
                </span>
                {x.r.why?.length ? <Chips values={x.r.why} /> : null}
              </span>
            ) : (
              '—'
            ),
        },
      ]}
    />
  );
}

function Tests({ x, busy, act }: { x: SkillDetail; busy: boolean; act: (path: string, body: unknown, ok: string) => Promise<unknown> }) {
  const [run, setRun] = useState<ActivationRun | null>(null);
  const [runMsg, setRunMsg] = useState<Msg>(null);
  const [edit, setEdit] = useState<string | null>(null);
  const [editErr, setEditErr] = useState<Msg>(null);
  const shown = x.versions.find((v) => v.version === (x.text_version ?? x.served_version)) ?? x.versions[0];
  const last = run ?? shown?.validate?.activation;
  const t: SkillTests = x.tests ?? {};
  const should = t.activation?.should ?? [];
  const shouldNot = t.activation?.should_not ?? [];
  const behaviour = t.behaviour ?? [];
  const runNow = async () => {
    setRunMsg(null);
    const r = await send<{ activation: ActivationRun }>(SKILL_POST.activation, { id: x.id }, 'ran against the armed pool now; no state change');
    setRunMsg(r.msg);
    if (r.data) setRun(r.data.activation);
  };
  return (
    <Panel
      kanji="試"
      title="ACTIVATION TESTS"
      tag={last ? `${last.passed ? 'PASSED' : 'FAILED'} · ${last.score === null ? '—' : `${Math.round(last.score * 100)}%`}` : 'NO RESULT'}
      tagTone={last ? (last.passed ? 'moss' : 'crimson') : 'muted'}
      sub={run ? `run now · pool ${run.pool?.size ?? '—'} armed · ${run.embedding ?? ''}` : last ? `last result, v${shown?.version} · ${last.embedding ?? ''}` : undefined}
    >
      <CaseRows cases={should} kind="should" run={last ?? undefined} />
      <CaseRows cases={shouldNot} kind="should_not" run={last ?? undefined} />
      <div {...stylex.props(layout.rowWrap)}>
        <Btn disabled={busy} onClick={() => void runNow()}>
          RUN NOW
        </Btn>
        <Btn disabled={busy || edit !== null} onClick={() => setEdit(JSON.stringify({ activation: t.activation ?? { should: [], should_not: [] }, behaviour }, null, 2))}>
          EDIT TESTS
        </Btn>
      </div>
      <Note msg={runMsg} />
      {edit !== null ? (
        <>
          <textarea aria-label="tests JSON" value={edit} onChange={(e) => setEdit(e.target.value)} {...stylex.props(text.bodySm, p.input, p.area, p.areaTall)} />
          <div {...stylex.props(layout.rowWrap)}>
            <Btn
              disabled={busy}
              onClick={() => {
                let tests: unknown;
                try {
                  tests = JSON.parse(edit);
                } catch (e) {
                  setEditErr({ tone: 'err', text: `not JSON: ${String(e)}` });
                  return;
                }
                setEditErr(null);
                void act(SKILL_POST.tests, { id: x.id, tests }, 'new tests saved as a new version; it re-validates before it re-arms').then((ok) => {
                  if (ok) setEdit(null);
                });
              }}
            >
              SAVE TESTS · RE-VALIDATE
            </Btn>
            <Btn onClick={() => setEdit(null)}>CANCEL</Btn>
          </div>
          <Note msg={editErr} />
        </>
      ) : null}
      <Label>BEHAVIOUR CHECKS · {behaviour.length} · STORED, NOT RUN (THE LIVE A/B)</Label>
      {behaviour.length ? (
        behaviour.map((b, i) => (
          <Row key={i}>
            <div {...stylex.props(s.item)}>
              <span {...stylex.props(text.bodySm, p.why)}>{b.prompt}</span>
              <span {...stylex.props(text.labelXs, text.outline)}>
                {b.check?.kind ?? 'check'} {b.check?.pattern ? `/${b.check.pattern}/` : ''}
              </span>
              {b.why ? <span {...stylex.props(text.labelXs, text.outline)}>{b.why}</span> : null}
            </div>
          </Row>
        ))
      ) : (
        <StateView kind="empty" title="no behaviour checks stored" />
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function Version({ v, stages, x }: { v: SkillVersion; stages: string[]; x: SkillDetail }) {
  const why = splitRemedy(v.reason);
  const jobs = v.version === x.latest_version ? x.jobs : [];
  return (
    <details open={v.version === x.latest_version}>
      <summary {...stylex.props(text.labelMd, p.summary)}>
        v{v.version} · {v.origin} · {v.state} at {v.stage}
        {v.version === x.served_version ? ' · SERVED' : ''}
      </summary>
      <div {...stylex.props(layout.stackSm)} style={{ marginTop: 6 }}>
        <StageRail stages={stages} version={v} jobs={jobs} />
        {why.situation ? (
          <div {...stylex.props(layout.stack)}>
            <p {...stylex.props(text.bodySm, v.state === 'failed' || v.state === 'quarantined' ? p.err : p.why)}>{why.situation}</p>
            {why.remedy ? <p {...stylex.props(text.bodySm, p.why)}>REMEDY {why.remedy}</p> : null}
          </div>
        ) : null}
        <span {...stylex.props(text.labelXs, text.outline)}>
          by {v.author ?? '—'} · created {ago(Date.now() / 1000 - v.created)} ago{v.armed_at ? ` · armed ${ago(Date.now() / 1000 - v.armed_at)} ago` : ''}
          {v.source_bytes ? ` · source ${n(v.source_bytes)} bytes` : ''}
        </span>
        {(v.screen?.deterministic?.quarantine ?? []).map((f, i) => (
          <Row key={`q${i}`} bad>
            <span {...stylex.props(p.err)}>
              {f.rule}: {f.what}
              {f.line ? ` (line ${f.line})` : ''}
            </span>
          </Row>
        ))}
        {(v.path ?? []).map((st) => (
          <Raw key={st} label={`${st} output`} value={stageOutput(v, st)} />
        ))}
        {v.text && v.version !== (x.text_version ?? -1) ? <Raw label="this version's SKILL.md" value={v.text} /> : null}
      </div>
    </details>
  );
}

function Versions({ x, stages }: { x: SkillDetail; stages: string[] }) {
  return (
    <Panel kanji="歴" title="REVISIONS" tag={`${x.versions.length}`} tagTone="muted" sub="every version with every stage's output">
      {x.versions.map((v) => (
        <Version key={v.version} v={v} stages={stages} x={x} />
      ))}
      {x.jobs.length ? (
        <details>
          <summary {...stylex.props(text.labelXs, p.summary)}>{x.jobs.length} job(s)</summary>
          <Table
            rows={x.jobs}
            rowKey={(j) => j.id}
            bad={(j) => !!j.error}
            columns={[
              { key: 'q', head: 'job', cell: (j) => j.queue },
              { key: 's', head: 'state', cell: (j) => j.state },
              { key: 'a', head: 'tries', num: true, cell: (j) => (j.attempts === undefined ? '—' : `${j.attempts}/${j.max_attempts ?? '—'}`) },
              { key: 'e', head: 'error / progress', cell: (j) => <span {...stylex.props(j.error ? p.err : p.why)}>{j.error ?? j.progress ?? '—'}</span> },
            ]}
          />
        </details>
      ) : null}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function Selections({ x }: { x: SkillDetail }) {
  return (
    <Panel kanji="録" title="SELECTED IN" tag={`${x.selections.length}`} tagTone="muted" sub="the most recent decisions that injected it; the request is a hash">
      {x.selections.length ? (
        <Table
          rows={x.selections}
          rowKey={(r, i) => `${r.ts}-${i}`}
          columns={[
            { key: 't', head: 'when', cell: (r) => `${ago(Date.now() / 1000 - r.ts)} ago` },
            { key: 'c', head: 'class', cell: (r) => r.route_class ?? '—' },
            { key: 'v', head: 'version', num: true, cell: (r) => (r.version === null ? '—' : `v${r.version}`) },
            { key: 'd', head: 'decided by', cell: (r) => `${r.decided_by ?? '—'} · ${r.strength ?? '—'}` },
            { key: 'r', head: 'request', cell: (r) => <span {...stylex.props(text.labelXs, text.outline)}>{r.request ?? '—'}</span> },
          ]}
        />
      ) : (
        <StateView kind="empty" title="not selected yet" />
      )}
    </Panel>
  );
}

function Children({ x }: { x: SkillDetail }) {
  const parent = typeof x.meta.parent === 'string' ? x.meta.parent : typeof x.provenance?.parent === 'string' ? x.provenance.parent : null;
  if (!x.children.length && !parent) return null;
  return (
    <Panel kanji="枝" title={x.children.length ? 'DECOMPOSED INTO' : 'DECOMPOSED FROM'} tag={x.children.length ? `${x.children.length}` : undefined} tagTone="cyan">
      {parent ? (
        <NavLink to={href.skill(parent)} {...stylex.props(p.link)}>
          the frontier source {parent}
        </NavLink>
      ) : null}
      {x.children.map((c) => (
        <Row key={c.id}>
          <NavLink to={href.skill(c.id)} {...stylex.props(p.link)}>
            {c.name}
          </NavLink>
          <Chip tone={statusTone(c.status)}>{c.status}</Chip>
        </Row>
      ))}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function Actions({ x, busy, msg, act }: { x: SkillDetail; busy: boolean; msg: Msg; act: (path: string, body: unknown, ok: string) => Promise<unknown> }) {
  const latest = x.versions[0];
  const path = latest?.path ?? [];
  const [reason, setReason] = useState('');
  const [stage, setStage] = useState('');
  const [spdx, setSpdx] = useState('');
  const [quote, setQuote] = useState('');
  const [hours, setHours] = useState('');
  const rerunOk = canRerun(latest);
  const st = stage || latest?.stage || path[0] || '';
  return (
    <Panel kanji="操" title="ACTIONS" tag="RECORDED AS YOURS" tagTone="muted">
      <Field label="reason (recorded; quarantine needs one)">
        <input value={reason} onChange={(e) => setReason(e.target.value)} {...stylex.props(text.bodySm, p.input)} />
      </Field>
      <div {...stylex.props(layout.rowWrap)}>
        {x.status === 'disabled' || x.status === 'archived' ? (
          <Btn disabled={busy} onClick={() => void act(SKILL_POST.enable, { id: x.id }, 'enabled: the served version serves again')}>
            ENABLE
          </Btn>
        ) : (
          <Btn disabled={busy || !x.enabled} onClick={() => void act(SKILL_POST.disable, { id: x.id, reason }, 'disabled')}>
            DISABLE
          </Btn>
        )}
        <Btn disabled={busy || x.status === 'archived'} onClick={() => void act(SKILL_POST.archive, { id: x.id, reason }, 'archived: out of service and out of the default view')}>
          ARCHIVE
        </Btn>
        <Btn danger disabled={busy || !reason.trim() || x.status === 'quarantined'} title="an operator's quarantine: the latest version is marked and the skill disarmed" onClick={() => void act(SKILL_POST.quarantine, { id: x.id, reason }, 'quarantined')}>
          QUARANTINE
        </Btn>
      </div>
      <div {...stylex.props(s.fields)}>
        <Field label={`re-run v${latest?.version ?? '—'} from`}>
          <select aria-label="stage to re-run from" value={st} disabled={!rerunOk} onChange={(e) => setStage(e.target.value)} {...stylex.props(text.bodySm, p.input)}>
            {path.map((x2) => (
              <option key={x2} value={x2}>
                {x2}
              </option>
            ))}
          </select>
        </Field>
        <Btn disabled={busy || !rerunOk || !st} title={rerunOk ? undefined : `v${latest?.version} is ${latest?.state}: edit to make a new version instead`} onClick={() => void act(SKILL_POST.rerun, { id: x.id, stage: st }, `re-running from ${st}`)}>
          RE-RUN
        </Btn>
      </div>
      {!rerunOk && latest ? <span {...stylex.props(text.labelXs, text.outline)}>{`v${latest.version} is ${latest.state}: re-run is for a version that stopped; edit to make a new one`}</span> : null}
      <div {...stylex.props(s.fields)}>
        <Field label="licence (SPDX)">
          <input value={spdx} onChange={(e) => setSpdx(e.target.value)} placeholder={licenceOf(x)} {...stylex.props(text.bodySm, p.input)} />
        </Field>
        <Field label="the text that grants it, verbatim">
          <input value={quote} onChange={(e) => setQuote(e.target.value)} {...stylex.props(text.bodySm, p.input)} />
        </Field>
        <Btn disabled={busy || !spdx.trim() || !quote.trim()} onClick={() => void act(SKILL_POST.licence, { id: x.id, licence: spdx, quote }, 'licence recorded; re-run the licence stage to use it')}>
          RECORD LICENCE
        </Btn>
      </div>
      {x.source_url ? (
        <div {...stylex.props(s.fields)}>
          <Field label="re-fetch every N hours (0 stops)">
            <input inputMode="decimal" value={hours} onChange={(e) => setHours(e.target.value)} placeholder={x.watch_seconds ? String(x.watch_seconds / 3600) : 'off'} {...stylex.props(text.bodySm, p.input)} />
          </Field>
          <Btn disabled={busy || hours.trim() === '' || !Number.isFinite(Number(hours))} onClick={() => void act(SKILL_POST.watch, { id: x.id, hours: Number(hours) }, 'watch set')}>
            SET WATCH
          </Btn>
          <Btn disabled={busy} onClick={() => void act(SKILL_POST.refetch, { id: x.id }, 're-fetch queued')}>
            RE-FETCH NOW
          </Btn>
        </div>
      ) : null}
      <Note msg={msg} />
    </Panel>
  );
}

// ---------------------------------------------------------------------------
export function SkillScreen({ id }: { id: string }) {
  const [every, setEvery] = useState(10000);
  const d = usePoll<{ skill: SkillDetail }>(skillPath(id), every);
  const o = usePoll<SkillsOverview>(SKILLS_PATH, 60000);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const x = d.data?.skill ?? null;
  // Faster while the pipeline is working on it.
  useEffect(() => setEvery(x?.status === 'pipeline' ? 3000 : 10000), [x?.status]);
  const act = async (path: string, body: unknown, ok: string): Promise<boolean> => {
    setBusy(true);
    const r = await send(path, body, ok);
    setBusy(false);
    setMsg(r.msg);
    if (r.data) {
      d.reload();
      o.reload();
    }
    return !!r.data;
  };
  const nav = <SubNav current={null} extra={<NavLink to={href.skills} {...stylex.props(text.labelXs, p.link)}>← LIBRARY</NavLink>} />;
  if (!x) {
    return (
      <div {...stylex.props(s.page)}>
        {nav}
        <PollState path={skillPath(id)} failure={d.failure} />
      </div>
    );
  }
  const l = o.data?.limits ?? null;
  const stages = o.data?.stages ?? x.versions[0]?.path ?? [];
  return (
    <div {...stylex.props(s.page)}>
      {nav}
      <div {...stylex.props(s.grid)}>
        <div {...stylex.props(s.col)}>
          <Header x={x} l={l} stale={d.stale} />
          <SkillMd x={x} busy={busy} onSave={(t) => void act(SKILL_POST.edit, { id: x.id, text: t }, 'saved as a new version; it re-arms only after the screen, its tests and validation')} />
          <Items x={x} l={l} />
          <Tests x={x} busy={busy} act={act} />
          <Versions x={x} stages={stages} />
        </div>
        <div {...stylex.props(s.col)}>
          <Actions x={x} busy={busy} msg={msg} act={act} />
          <Taxonomy x={x} o={o.data} />
          <Children x={x} />
          <Selections x={x} />
          {!l ? <PollState path={SKILLS_PATH} failure={o.failure} /> : null}
        </div>
      </div>
    </div>
  );
}
