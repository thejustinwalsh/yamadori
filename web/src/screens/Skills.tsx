// 技 SKILLS — the skill library that replaced hints. Every skill, what it
// applies to, whether it is armed, and why not. The pipeline arms a skill
// with no review; a person reviews here whenever they like: edit (a new
// version, re-screened before it re-arms), disable, re-enable, add.
import * as stylex from '@stylexjs/stylex';
import { useState } from 'react';
import { postJson } from '../api/client';
import {
  appliesToText,
  createBody,
  overallFallbackRate,
  SKILL_POST,
  SKILLS_PATH,
  skillPath,
  statusTone,
  versionText,
  type SkillDetail,
  type SkillsOverview,
  type SkillSummary,
} from '../api/skills';
import { usePoll } from '../api/usePoll';
import { ago, n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Row, Stat } from '../ui/primitives';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';

const s = stylex.create({
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'minmax(0, 3fr) minmax(0, 2fr)' },
    alignItems: 'start',
  },
  col: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  input: {
    boxSizing: 'border-box',
    width: '100%',
    minWidth: 0,
    backgroundColor: colors.surfaceContainerLowest,
    color: colors.tertiaryContainer,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.outlineVariant,
    paddingInline: space.spaceSm,
    paddingBlock: space.spaceXs,
    outline: { default: 'none', ':focus-visible': `1px solid ${colors.tertiaryContainer}` },
  },
  area: { minHeight: '10rem', resize: 'vertical', fontFamily: 'inherit' },
  button: {
    backgroundColor: { default: colors.surfaceContainerLowest, ':hover': `color-mix(in srgb, ${colors.primaryContainer} 15%, transparent)` },
    color: colors.primaryContainer,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.primaryContainer,
    paddingInline: space.spaceMd,
    paddingBlock: space.spaceXs,
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { default: 1, ':disabled': 0.5 },
    whiteSpace: 'nowrap',
  },
  pick: { background: 'none', borderWidth: 0, padding: 0, color: colors.tertiaryContainer, cursor: 'pointer', textAlign: 'left' },
  pre: { margin: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', color: colors.onSurface },
  why: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere', textTransform: 'none' },
  err: { margin: 0, color: colors.secondary, overflowWrap: 'anywhere' },
});

type Msg = { tone: 'ok' | 'err'; text: string } | null;

function Note({ msg }: { msg: Msg }) {
  if (!msg) return null;
  return (
    <p role={msg.tone === 'err' ? 'alert' : 'status'} {...stylex.props(text.bodySm, msg.tone === 'err' ? s.err : s.why)}>
      {msg.tone === 'err' ? '[ ! ] ' : ''}
      {msg.text}
    </p>
  );
}

async function send(path: string, body: unknown): Promise<Msg> {
  const r = await postJson<{ ok: true }>(path, body);
  if (r.ok) return { tone: 'ok', text: 'saved' };
  const f = r.failure;
  return { tone: 'err', text: 'message' in f ? f.message : f.kind };
}

/** The skill list. Presentational, so it renders in a test. */
export function SkillList({ skills, onPick }: { skills: SkillSummary[]; onPick: (id: string) => void }) {
  if (!skills.length) return <StateView kind="empty" title="no skills" detail="add a URL or paste a skill below, or run mcp/skill_migrate.py" />;
  return (
    <Table
      tall
      rows={skills}
      rowKey={(x) => x.id}
      bad={(x) => x.status === 'quarantined' || x.status === 'failed'}
      columns={[
        {
          key: 't',
          head: 'skill',
          cell: (x) => (
            <button type="button" onClick={() => onPick(x.id)} {...stylex.props(text.bodySm, s.pick)}>
              {x.title || x.name}
            </button>
          ),
        },
        { key: 's', head: 'status', cell: (x) => <Chip tone={statusTone(x.status)}>{x.status}</Chip> },
        { key: 'a', head: 'applies to', cell: (x) => appliesToText(x.applies_to) },
        { key: 'v', head: 'served', cell: (x) => versionText(x) },
        { key: 'u', head: 'updated', cell: (x) => `${ago(Date.now() / 1000 - x.updated)} ago` },
      ]}
    />
  );
}

function Detail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const d = usePoll<{ skill: SkillDetail }>(skillPath(id), 10000);
  const [draft, setDraft] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  if (!d.data) return <PollState path={skillPath(id)} failure={d.failure} />;
  const x = d.data.skill;
  const act = async (path: string, body: unknown) => {
    setBusy(true);
    const m = await send(path, body);
    setBusy(false);
    setMsg(m);
    if (m?.tone === 'ok') {
      setDraft(null);
      d.reload();
      onChanged();
    }
  };
  const latest = x.versions[0];
  const findings = latest?.screen?.deterministic?.quarantine ?? [];
  return (
    <Panel kanji="技" title={x.title || x.name} tag={x.status.toUpperCase()} tagTone={statusTone(x.status)} flag={x.id} stale={d.stale} edge={statusTone(x.status)}>
      {x.reason ? <p {...stylex.props(text.bodySm, x.status === 'armed' ? s.why : s.err)}>{x.reason}</p> : null}
      <div {...stylex.props(layout.grid3)}>
        <Stat label="served" value={versionText(x)} />
        <Stat label="applies to" value={appliesToText(x.applies_to)} />
        <Stat label="watch" value={x.watch_seconds ? `every ${n(x.watch_seconds / 3600)} h` : 'off'} />
      </div>
      <Label>APPLIES WHEN · {x.applies_when ?? '—'}</Label>
      {x.triggers.length || x.learned_triggers.length ? (
        <div {...stylex.props(layout.stack)}>
          <Label>TRIGGERS</Label>
          {x.triggers.map((t, i) => (
            <Row key={`t${i}`}>
              <span {...stylex.props(s.why)}>{t}</span>
            </Row>
          ))}
          {x.learned_triggers.map((t, i) => (
            <Row key={`l${i}`}>
              <span {...stylex.props(s.why)}>{t}</span>
              <Chip tone="cyan">learned</Chip>
            </Row>
          ))}
        </div>
      ) : null}
      {draft === null ? (
        x.text ? <pre {...stylex.props(text.bodySm, s.pre)}>{x.text}</pre> : <StateView kind="inert" title="no validated text yet" />
      ) : (
        <textarea aria-label="skill text" value={draft} onChange={(e) => setDraft(e.target.value)} {...stylex.props(text.bodySm, s.input, s.area)} />
      )}
      {findings.length ? (
        <div {...stylex.props(layout.stack)}>
          <Label>SCREEN FINDINGS · v{latest?.version}</Label>
          {findings.map((f, i) => (
            <Row key={i} bad>
              <span {...stylex.props(s.err)}>
                {f.rule}: {f.what}
                {f.line ? ` (line ${f.line})` : ''}
              </span>
            </Row>
          ))}
        </div>
      ) : null}
      <div {...stylex.props(layout.rowWrap)}>
        {draft === null ? (
          <button type="button" disabled={busy} onClick={() => setDraft(x.text ?? `${x.title ?? ''}\napplies when: ${x.applies_when ?? ''}\n- DO: `)} {...stylex.props(text.labelMd, s.button)}>
            [ EDIT ]
          </button>
        ) : (
          <>
            <button type="button" disabled={busy || !draft.trim()} onClick={() => void act(SKILL_POST.edit, { id: x.id, text: draft })} {...stylex.props(text.labelMd, s.button)}>
              [ SAVE · RE-SCREEN ]
            </button>
            <button type="button" disabled={busy} onClick={() => setDraft(null)} {...stylex.props(text.labelMd, s.button)}>
              [ CANCEL ]
            </button>
          </>
        )}
        {x.enabled ? (
          <button type="button" disabled={busy} onClick={() => void act(SKILL_POST.disable, { id: x.id })} {...stylex.props(text.labelMd, s.button)}>
            [ DISABLE ]
          </button>
        ) : (
          <button type="button" disabled={busy} onClick={() => void act(SKILL_POST.enable, { id: x.id })} {...stylex.props(text.labelMd, s.button)}>
            [ RE-ENABLE ]
          </button>
        )}
        {x.source_url ? (
          <button type="button" disabled={busy} onClick={() => void act(SKILL_POST.refetch, { id: x.id })} {...stylex.props(text.labelMd, s.button)}>
            [ RE-FETCH ]
          </button>
        ) : null}
      </div>
      <Note msg={msg} />
      <Table
        rows={x.versions}
        rowKey={(v) => String(v.version)}
        bad={(v) => v.state === 'quarantined' || v.state === 'failed'}
        columns={[
          { key: 'v', head: 'version', num: true, cell: (v) => `v${v.version}` },
          { key: 'o', head: 'origin', cell: (v) => `${v.origin}${v.author ? ` · ${v.author}` : ''}` },
          { key: 's', head: 'state', cell: (v) => `${v.state} at ${v.stage}` },
          { key: 'k', head: 'items kept', num: true, cell: (v) => (v.validate?.counts ? `${v.validate.counts.kept ?? 0}/${v.validate.counts.proposed ?? 0}` : '—') },
          { key: 'r', head: 'reason', cell: (v) => <span {...stylex.props(v.reason ? s.err : s.why)}>{v.reason ?? '—'}</span> },
        ]}
      />
    </Panel>
  );
}

function AddSkill({ onAdded }: { onAdded: () => void }) {
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  return (
    <Panel kanji="加" title="ADD A SKILL" tag="ARMS WITHOUT REVIEW" tagTone="cyan">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setBusy(true);
          void send(SKILL_POST.create, createBody(input)).then((m) => {
            setBusy(false);
            setMsg(m);
            if (m?.tone === 'ok') {
              setInput('');
              onAdded();
            }
          });
        }}
        {...stylex.props(layout.stackSm)}
      >
        <textarea aria-label="skill URL or text" placeholder="a URL (a SKILL.md, a docs page) or the skill's text" value={input} onChange={(e) => setInput(e.target.value)} {...stylex.props(text.bodySm, s.input, s.area)} />
        <div {...stylex.props(layout.rowWrap)}>
          <button type="submit" disabled={busy || !input.trim()} {...stylex.props(text.labelMd, s.button)}>
            [ INGEST ]
          </button>
        </div>
        <p {...stylex.props(text.bodySm, s.why)}>fetch → screen → model screen → classify → distil → validate → arm. A failed screen quarantines it and says why.</p>
        <Note msg={msg} />
      </form>
    </Panel>
  );
}

function Learning({ o }: { o: SkillsOverview }) {
  const rate = overallFallbackRate(o.learning.rate);
  return (
    <Panel kanji="学" title="SELECTION · FALLBACK RATE" tag={o.recall === 'skills' ? 'SKILLS PATH ON' : 'HINTS PATH (DEFAULT)'} tagTone={o.recall === 'skills' ? 'moss' : 'muted'}>
      <div {...stylex.props(layout.grid3)}>
        <Stat label="fallback rate" value={rate === null ? '—' : `${Math.round(rate * 100)}%`} sub={`${o.learning.rate.length} day(s)`} />
        <Stat label="pending records" value={n(o.learning.pending)} />
        <Stat label="armed" value={n(o.counts.armed ?? 0)} />
      </div>
      <p {...stylex.props(text.bodySm, s.why)}>learning: {o.learning.idle.why}</p>
      {o.learning.rate.length ? (
        <Table
          rows={o.learning.rate}
          rowKey={(d) => d.day}
          columns={[
            { key: 'd', head: 'day', cell: (d) => d.day },
            { key: 'c', head: 'with candidates', num: true, cell: (d) => n(d.with_candidates) },
            { key: 'i', head: 'injected', num: true, cell: (d) => n(d.injected) },
            { key: 'f', head: 'fallbacks', num: true, cell: (d) => n(d.fallbacks) },
            { key: 'r', head: 'rate', num: true, cell: (d) => (d.fallback_rate === null ? '—' : `${Math.round(d.fallback_rate * 100)}%`) },
          ]}
        />
      ) : (
        <StateView kind="empty" title="no selections recorded yet" />
      )}
      <Label>LIMITS · {o.limits.label}</Label>
    </Panel>
  );
}

export function Skills() {
  const o = usePoll<SkillsOverview>(SKILLS_PATH, 10000);
  const [picked, setPicked] = useState<string | null>(null);
  if (!o.data) return <PollState path={SKILLS_PATH} failure={o.failure} />;
  const data = o.data;
  return (
    <div {...stylex.props(s.grid)}>
      <div {...stylex.props(s.col)}>
        <Panel kanji="技" title="SKILLS" tag={`${data.skills.length}`} tagTone="cyan" stale={o.stale} sub={Object.entries(data.counts).map(([k, v]) => `${k} ${v}`).join(' · ')}>
          <SkillList skills={data.skills} onPick={setPicked} />
        </Panel>
        <AddSkill onAdded={o.reload} />
      </div>
      <div {...stylex.props(s.col)}>
        {picked ? <Detail id={picked} onChanged={o.reload} /> : <StateView kind="inert" title="pick a skill to see its versions and edit it" />}
        <Learning o={data} />
      </div>
    </div>
  );
}
