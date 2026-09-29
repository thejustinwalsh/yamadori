// 技 One package onboarding (docs/PACKAGE-ONBOARDING.md 8.2): one panel per
// stage with its result -- the resolution, the licence quotes, the index,
// the vocabulary and what existing packages lose, the examples, k and its
// curve, the sources by tier, the skills, the retire record and the
// EVALUATION, every number with its n -- and a REVIEW note box on each.
// Review is after the fact and changes nothing (no review gate). Everything
// from GET /dash/api/skill-factory/onboarding/<id>.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useState, type ReactNode } from 'react';
import {
  canForce,
  curveRows,
  DATASET_POST,
  evalView,
  lostNames,
  ONBOARDING_POST,
  onboardingPath,
  onboardingTone,
  packageLabel,
  RAIL_TONE,
  railCells,
  scalars,
  stageRecord,
  tier3Pending,
  waitingLine,
  type OnboardingDetail,
  type RailCell,
} from '../api/onboarding';
import { statusTone } from '../api/skills';
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
import { Blockers, ClarifyForms, ErroredJobs, OnboardingRail, useAct } from './skills/onboarding';
import { Btn, Chips, Note, Raw, s as p, SubNav } from './skills/parts';

const s = stylex.create({
  page: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'repeat(2, minmax(0, 1fr))' },
    alignItems: 'start',
  },
  kv: { display: 'grid', gridTemplateColumns: 'minmax(7rem, auto) minmax(0, 1fr)', gap: space.spaceXs, alignItems: 'start' },
  val: { color: 'inherit', overflowWrap: 'anywhere', textTransform: 'none' },
  banner: { margin: 0, textTransform: 'none' },
});

type Act = (path: string, body: unknown, ok: string) => Promise<boolean>;

function KV({ rows }: { rows: [string, ReactNode][] }) {
  if (!rows.length) return null;
  return (
    <div {...stylex.props(s.kv)}>
      {rows.map(([k, v]) => (
        <Row key={k}>
          <Label>{k}</Label>
          <span {...stylex.props(text.bodySm, s.val)}>{v}</span>
        </Row>
      ))}
    </div>
  );
}

const STAGE_KANJI: Record<string, string> = {
  resolve: '解',
  clarify: '明',
  index: '索',
  vocab: '語',
  examples: '例',
  knn: '近',
  sources: '源',
  skills: '技',
  retire: '退',
  rebuild: '築',
  evaluate: '評',
};

// ---------------------------------------------------------------------------
// The review box: an after-the-fact note on one stage, and the notes so far.
// ---------------------------------------------------------------------------
function ReviewBox({ d, stage, busy, act }: { d: OnboardingDetail; stage: string; busy: boolean; act: Act }) {
  const [note, setNote] = useState('');
  const mine = d.reviews.filter((r) => r.stage === stage);
  return (
    <details>
      <summary {...stylex.props(text.labelXs, p.summary)}>REVIEW{mine.length ? ` · ${mine.length} note(s)` : ''} · changes nothing</summary>
      <div {...stylex.props(layout.stackSm)} style={{ marginTop: 6 }}>
        {mine.map((r, i) => (
          <Row key={i}>
            <span {...stylex.props(p.why)}>{r.note}</span>
            <span {...stylex.props(text.labelXs, text.outline)}>
              {r.by} · {ago(Date.now() / 1000 - r.at)} ago
            </span>
          </Row>
        ))}
        <textarea aria-label={`review note on ${stage}`} value={note} onChange={(e) => setNote(e.target.value)} placeholder={`what you saw in ${stage}`} {...stylex.props(text.bodySm, p.input, p.area)} />
        <div {...stylex.props(layout.rowWrap)}>
          <Btn
            disabled={busy || !note.trim()}
            onClick={() =>
              void act(ONBOARDING_POST.review, { id: d.id, stage, note: note.trim() }, `note on ${stage} saved`).then((ok) => {
                if (ok) setNote('');
              })
            }
          >
            SAVE NOTE
          </Btn>
        </div>
      </div>
    </details>
  );
}

// ---------------------------------------------------------------------------
// Each stage's result
// ---------------------------------------------------------------------------
function Resolve({ d }: { d: OnboardingDetail }) {
  const r = d.resolution;
  if (!r) return null;
  const pkg = r.package;
  const links = r.links ?? [];
  const linkRows = links.map((l) => {
    if (pkg?.links?.includes(l)) return { l, to: packageLabel({ package: pkg.name, version: pkg.version ?? null, id: d.id }), how: `version: ${pkg.version_rule ?? '—'}`, bad: false };
    const u = (r.unresolved ?? []).find((x) => x.link === l);
    if (u) return { l, to: u.name ? `unresolved: ${u.name}` : 'unresolved', how: u.why ?? '—', bad: true };
    const src = (r.sources ?? []).find((x) => x.url === l);
    if (src) return { l, to: `a skill source (${src.kind ?? 'page'})`, how: src.attach_to ? `attached to ${src.attach_to}` : 'attached to the first package', bad: false };
    return { l, to: 'another package of the group (its own onboarding), or not a package link', how: '—', bad: false };
  });
  const rep = r.replaces;
  return (
    <>
      <Table
        rows={linkRows}
        rowKey={(x) => x.l}
        bad={(x) => x.bad}
        columns={[
          { key: 'l', head: 'link', cell: (x) => <span {...stylex.props(text.labelXs, s.val)}>{x.l}</span> },
          { key: 't', head: 'resolved to', cell: (x) => <span {...stylex.props(s.val)}>{x.to}</span> },
          { key: 'h', head: 'the rule', cell: (x) => <span {...stylex.props(p.why)}>{x.how}</span> },
        ]}
      />
      {pkg ? (
        <Table
          rows={[
            { k: 'package', v: `${pkg.ecosystem ?? ''} ${pkg.name}`.trim(), rule: pkg.unsupported ?? '', why: '' },
            { k: 'version', v: pkg.version ?? '—', rule: pkg.version_rule ?? '—', why: pkg.version_why ?? '' },
            { k: 'commit', v: pkg.commit ? pkg.commit.slice(0, 12) : '—', rule: pkg.commit_rule ?? '—', why: pkg.repository ? `${pkg.repository.owner}/${pkg.repository.repo}${pkg.repository.directory ? `/${pkg.repository.directory}` : ''}` : '' },
            { k: 'published', v: pkg.published ?? '—', rule: '', why: '' },
            { k: 'tarball', v: pkg.integrity ? String(pkg.integrity).slice(0, 24) + '…' : '—', rule: pkg.tarball ? 'dist.integrity' : '', why: pkg.tarball ?? '' },
          ]}
          rowKey={(x) => x.k}
          columns={[
            { key: 'k', head: 'what', cell: (x) => x.k },
            { key: 'v', head: 'value', cell: (x) => <span {...stylex.props(s.val)}>{x.v}</span> },
            { key: 'r', head: 'chosen by', cell: (x) => <span {...stylex.props(p.why)}>{x.rule || '—'}</span> },
            { key: 'w', head: 'why / where', cell: (x) => <span {...stylex.props(p.why)}>{x.why || '—'}</span> },
          ]}
        />
      ) : (
        <p {...stylex.props(text.bodySm, p.why)}>no package: the links run as skill sources</p>
      )}
      <KV
        rows={[
          ...(r.how ? ([['how', r.how]] as [string, ReactNode][]) : []),
          ...(rep ? ([['replaces', `${rep.mode}${rep.old?.length ? ` ${rep.old.join(', ')}` : ''} · ${rep.why ?? ''}`]] as [string, ReactNode][]) : []),
          ...(rep?.risk ? ([['risk', rep.risk]] as [string, ReactNode][]) : []),
          ...(r.group ? ([['group', `${r.group}${r.group_size ? ` · ${r.group_size} package(s)` : ''}`]] as [string, ReactNode][]) : []),
        ]}
      />
      {r.siblings?.length ? (
        <div {...stylex.props(layout.stack)}>
          <Label>SIBLINGS (ONE ONBOARDING PER PACKAGE)</Label>
          {r.siblings.map((x) => (
            <NavLink key={x.id} to={href.onboarding(x.id)} {...stylex.props(p.link)}>
              {x.package ?? x.id}
            </NavLink>
          ))}
        </div>
      ) : null}
    </>
  );
}

function Licence({ d, busy, act }: { d: OnboardingDetail; busy: boolean; act: Act }) {
  const l = d.licence;
  const quotes = l?.quotes ?? [];
  return (
    <>
      <KV
        rows={[
          ['licence', d.licence_value ?? 'not established'],
          ...(l?.chosen ? ([['chosen', `${l.chosen.spdx} · ${l.chosen.where ?? ''}`]] as [string, ReactNode][]) : []),
          ...(l && typeof l.agree === 'boolean' ? ([['sources agree', l.agree ? 'yes' : 'NO']] as [string, ReactNode][]) : []),
        ]}
      />
      {quotes.map((q, i) => (
        <div key={i} {...stylex.props(layout.stack)}>
          <span {...stylex.props(text.labelXs, text.outline)}>
            {q.spdx} · {q.kind ?? 'quote'} · {q.where ?? '—'}
          </span>
          <p {...stylex.props(text.bodySm, p.quote)}>“{q.quote}”</p>
        </div>
      ))}
      {l && !quotes.length ? <p {...stylex.props(text.bodySm, p.why)}>no verbatim licence quote was found</p> : null}
      <ClarifyForms d={d} busy={busy} act={act} />
    </>
  );
}

function Index({ d }: { d: OnboardingDetail }) {
  const x = d.index;
  if (!x) return null;
  return (
    <>
      {x.skipped ? <p {...stylex.props(text.bodySm, p.why)}>skipped: {x.skipped}</p> : null}
      {x.fetch ? (
        <>
          <Label>FETCH</Label>
          <KV rows={scalars(x.fetch)} />
        </>
      ) : null}
      {x.health ? (
        <>
          <Label>HEALTH</Label>
          <KV rows={scalars(x.health)} />
        </>
      ) : null}
    </>
  );
}

function Vocab({ d, busy, act }: { d: OnboardingDetail; busy: boolean; act: Act }) {
  const v = d.vocab;
  if (!v) return null;
  const per = Object.entries(v.diff?.per_package ?? {});
  const lost = lostNames(v);
  const gained = Object.entries(v.diff?.gained ?? {}).filter(([, names]) => names?.length);
  const fl = v.floor;
  return (
    <>
      <div {...stylex.props(layout.rowWrap)}>
        <Chip tone={v.state === 'held' ? 'crimson' : v.state === 'forced' ? 'rose' : v.state === 'promoted' ? 'moss' : 'muted'}>{v.state ?? 'not built'}</Chip>
        {v.forced_by ? <span {...stylex.props(text.labelXs, text.outline)}>forced by {v.forced_by}</span> : null}
      </div>
      {v.why ? <p {...stylex.props(text.bodySm, v.state === 'held' ? p.err : p.why)}>{v.why}</p> : null}
      {per.length ? (
        <Table
          rows={per}
          rowKey={([k]) => k}
          columns={[
            { key: 'p', head: 'package', cell: ([k]) => <span {...stylex.props(p.asIs)}>{k}</span> },
            ...['names', 'unique', 'code_only', 'dropped_shared', 'dropped_platform', 'dropped_english'].map((f) => ({
              key: f,
              head: f.replace('_', ' '),
              num: true,
              cell: ([, row]: [string, Record<string, unknown>]) => {
                const val = row[f];
                return typeof val === 'number' ? n(val) : Array.isArray(val) ? n(val.length) : '—';
              },
            })),
          ]}
        />
      ) : null}
      <Label>NAMES EXISTING PACKAGES LOSE · {lost.reduce((a, x) => a + x.names.length, 0)}</Label>
      {lost.length ? (
        lost.map((x) => (
          <Row key={x.package} bad>
            <span {...stylex.props(p.asIs)}>{x.package}</span>
            <Chips values={x.names} tone="rose" />
          </Row>
        ))
      ) : (
        <p {...stylex.props(text.bodySm, p.why)}>none</p>
      )}
      {gained.length ? (
        <details>
          <summary {...stylex.props(text.labelXs, p.summary)}>names existing packages gain · {gained.length} package(s)</summary>
          {gained.map(([pkg, names]) => (
            <Row key={pkg}>
              <span {...stylex.props(p.asIs)}>{pkg}</span>
              <Chips values={names} />
            </Row>
          ))}
        </details>
      ) : null}
      {fl ? (
        <>
          <Label>
            PROMOTION FLOOR · {fl.passed ? 'PASSED' : 'NOT PASSED'} · n = {fl.rows ?? '—'} LABEL ROWS
          </Label>
          {[...(fl.lost_tp ?? []).map((x) => ({ ...x, kind: 'lost true positive' })), ...(fl.gained_fp ?? []).map((x) => ({ ...x, kind: 'gained false positive' }))].map((x, i) => (
            <Row key={i} bad>
              <span {...stylex.props(p.err)}>
                {x.kind}: {x.package} on {x.id}
              </span>
            </Row>
          ))}
        </>
      ) : null}
      {canForce(v) ? (
        <div {...stylex.props(layout.rowWrap)}>
          <Btn danger disabled={busy} title="promote the HELD vocabulary anyway; recorded with you as the author" onClick={() => void act(ONBOARDING_POST.promote, { id: d.id }, 'vocabulary forced; the onboarding moves on')}>
            FORCE THE VOCABULARY
          </Btn>
        </div>
      ) : null}
    </>
  );
}

function Examples({ d }: { d: OnboardingDetail }) {
  const x = d.examples;
  if (!x) return null;
  const per = x.per_package && typeof x.per_package === 'object' && !Array.isArray(x.per_package) ? Object.entries(x.per_package as Record<string, unknown>) : [];
  return (
    <>
      {typeof x.skipped === 'string' ? <p {...stylex.props(text.bodySm, p.why)}>skipped: {x.skipped}</p> : null}
      <KV rows={scalars(x, ['skipped'])} />
      {per.length ? (
        <Table
          rows={per}
          rowKey={([k]) => k}
          columns={[
            { key: 'p', head: 'package', cell: ([k]) => <span {...stylex.props(p.asIs)}>{k}</span> },
            {
              key: 'v',
              head: 'counts',
              cell: ([, v]) =>
                v && typeof v === 'object' ? (
                  <span {...stylex.props(p.why)}>
                    {scalars(v as Record<string, unknown>)
                      .map(([a, b]) => `${a} ${b}`)
                      .join(' · ')}
                  </span>
                ) : (
                  String(v)
                ),
            },
          ]}
        />
      ) : null}
      {Object.entries(x)
        .filter(([k, v]) => k !== 'per_package' && v && typeof v === 'object')
        .map(([k, v]) => (
          <Raw key={k} label={k} value={v} />
        ))}
    </>
  );
}

function Knn({ d }: { d: OnboardingDetail }) {
  const x = d.knn;
  if (!x) return null;
  const curve = curveRows(x.curve);
  return (
    <>
      {x.skipped ? <p {...stylex.props(text.bodySm, p.why)}>skipped: {x.skipped}</p> : null}
      {x.k !== undefined ? (
        <div {...stylex.props(layout.grid3)}>
          <Stat label="k (measured)" value={x.k ?? '—'} sub={`n = ${x.k_n ?? '—'}`} />
          <Stat label="groups" value={x.groups ?? '—'} />
          <Stat label="curve points" value={curve.length} />
        </div>
      ) : null}
      <KV rows={scalars(x, ['k', 'k_n', 'groups', 'skipped'])} />
      {curve.length ? (
        <Table
          rows={curve}
          rowKey={(c) => c.k}
          columns={[
            { key: 'k', head: 'k', num: true, cell: (c) => c.k },
            { key: 'v', head: 'top-1 (leave one group out)', num: true, cell: (c) => <span {...stylex.props(c.k === String(x.k) ? text.moss : text.onSurface)}>{c.value}</span> },
            { key: 'n', head: 'n', num: true, cell: (c) => (c.n === null ? (x.k_n !== undefined ? `${x.k_n} (k's n)` : '—') : n(c.n)) },
          ]}
        />
      ) : null}
      {Object.entries(x)
        .filter(([k, v]) => k !== 'curve' && v && typeof v === 'object')
        .map(([k, v]) => (
          <Raw key={k} label={k} value={v} />
        ))}
    </>
  );
}

function Sources({ d, busy, act }: { d: OnboardingDetail; busy: boolean; act: Act }) {
  const x = d.sources;
  if (!x) return null;
  const chosen = [...(x.chosen ?? [])].sort((a, b) => a.tier - b.tier);
  const pending = tier3Pending(x);
  const t3 = x.tier3;
  const made = x.made;
  return (
    <>
      {chosen.length ? (
        <Table
          rows={chosen}
          rowKey={(c, i) => `${c.tier}-${c.url ?? c.path ?? i}`}
          columns={[
            { key: 't', head: 'tier', num: true, cell: (c) => c.tier },
            { key: 'u', head: 'source', cell: (c) => <span {...stylex.props(text.labelXs, s.val)}>{c.path ?? c.url ?? '—'}</span> },
            { key: 'f', head: 'path', cell: (c) => (c.frontier ? 'frontier' : 'distil') },
            { key: 'w', head: 'why', cell: (c) => <span {...stylex.props(p.why)}>{c.why ?? '—'}</span> },
          ]}
        />
      ) : (
        <p {...stylex.props(text.bodySm, p.why)}>no source chosen</p>
      )}
      {x.skipped?.length ? (
        <details>
          <summary {...stylex.props(text.labelXs, p.summary)}>{x.skipped.length} skipped</summary>
          {x.skipped.map((k, i) => (
            <Row key={i}>
              <span {...stylex.props(p.why)}>{k.path ?? k.url ?? '—'}</span>
              <span {...stylex.props(p.why)}>{k.why ?? ''}</span>
            </Row>
          ))}
        </details>
      ) : null}
      {t3 ? (
        <div {...stylex.props(layout.stack)}>
          <Label>
            TIER 3 · llms.txt · {t3.pages?.length ?? 0} page(s){t3.where ? ` · ${t3.where}` : ''}
            {t3.ingested_by ? ` · ingested by ${t3.ingested_by}` : ''}
          </Label>
          {t3.note ? <p {...stylex.props(text.bodySm, p.why)}>{t3.note}</p> : null}
          {pending ? (
            <div {...stylex.props(layout.rowWrap)}>
              <Btn disabled={busy} title="make a skill of each recorded page" onClick={() => void act(ONBOARDING_POST.tier3, { id: d.id }, 'tier-3 pages submitted as skills')}>
                {`INGEST ${pending} TIER-3 PAGE(S)`}
              </Btn>
            </div>
          ) : null}
        </div>
      ) : null}
      {made ? (
        <KV
          rows={Object.entries(made)
            .filter(([, v]) => Array.isArray(v))
            .map(([k, v]) => [k.replace('_', ' '), n((v as unknown[]).length)] as [string, ReactNode])}
        />
      ) : null}
      {x.examples_dirs?.length ? <KV rows={[['examples dirs', x.examples_dirs.join(', ')]]} /> : null}
    </>
  );
}

function Skills({ d }: { d: OnboardingDetail }) {
  if (!d.skills.length) return <StateView kind="inert" title="no skill made yet" />;
  return (
    <>
      <Label>
        {Object.entries(d.skill_counts)
          .filter(([, v]) => v)
          .map(([k, v]) => `${k} ${v}`)
          .join(' · ')}
      </Label>
      <Table
        rows={d.skills}
        rowKey={(k) => k.id}
        bad={(k) => k.status === 'failed' || k.status === 'quarantined'}
        columns={[
          {
            key: 'n',
            head: 'skill',
            cell: (k) => (
              <NavLink to={href.skill(k.id)} {...stylex.props(p.link)}>
                {k.name}
              </NavLink>
            ),
          },
          { key: 's', head: 'state', cell: (k) => <Chip tone={statusTone(k.status)}>{k.status}</Chip> },
          { key: 'v', head: 'served', cell: (k) => (k.version === null ? `none of ${k.latest_version ?? '—'}` : `v${k.version} of ${k.latest_version ?? '—'}`) },
          { key: 'l', head: 'lead for', cell: (k) => k.lead_for ?? '—' },
          {
            key: 'p',
            head: 'from',
            cell: (k) =>
              k.parent ? (
                <NavLink to={href.skill(k.parent)} {...stylex.props(p.link)}>
                  {k.parent}
                </NavLink>
              ) : (
                <span {...stylex.props(text.labelXs, s.val)}>{k.source_url ?? '—'}</span>
              ),
          },
          { key: 'r', head: 'why', cell: (k) => <span {...stylex.props(p.why)}>{k.reason ?? '—'}</span> },
        ]}
      />
    </>
  );
}

function Retire({ d }: { d: OnboardingDetail }) {
  const x = d.retire;
  if (!x) return null;
  return (
    <>
      <Label>ARCHIVED · {x.archived?.length ?? 0}</Label>
      {(x.archived ?? []).map((a) => (
        <Row key={a.id}>
          <NavLink to={href.skill(a.id)} {...stylex.props(p.link)}>
            {a.name ?? a.id}
          </NavLink>
          <span {...stylex.props(p.why)}>{a.why ?? ''}</span>
        </Row>
      ))}
      {(x.kept ?? []).map((a) => (
        <Row key={a.id}>
          <span>{a.id}</span>
          <span {...stylex.props(p.why)}>kept: {a.why ?? ''}</span>
        </Row>
      ))}
      <Raw label="the old version's vocabulary" value={x.vocab} />
    </>
  );
}

function Evaluation({ d }: { d: OnboardingDetail }) {
  const ev = evalView(d.eval);
  if (!d.eval) return null;
  const noN = ev.rows.filter((r) => r.num && r.n === null).length;
  return (
    <>
      <p {...stylex.props(text.labelMd, s.banner, ev.inSample ? text.rose : ev.inSample === false ? text.moss : text.outline)}>
        {ev.inSample === true ? 'IN-SAMPLE' : ev.inSample === false ? 'HELD OUT (not in-sample)' : 'in_sample NOT STATED'}
        {ev.inSampleWhy ? ` · ${ev.inSampleWhy}` : ''}
      </p>
      {noN ? <p {...stylex.props(text.bodySm, p.err)}>{noN} number(s) carry no n</p> : null}
      <Table
        tall
        rows={ev.rows}
        rowKey={(r, i) => `${r.where}.${r.measure}.${i}`}
        bad={(r) => r.num && r.n === null}
        columns={[
          { key: 'w', head: 'where', cell: (r) => <span {...stylex.props(text.labelXs, s.val)}>{r.where}</span> },
          { key: 'm', head: 'measure', cell: (r) => <span {...stylex.props(p.asIs)}>{r.measure}</span> },
          { key: 'v', head: 'value', num: true, cell: (r) => <span {...stylex.props(s.val)}>{r.value}</span> },
          { key: 'n', head: 'n', num: true, cell: (r) => (r.n === null ? (r.num ? <span {...stylex.props(text.rose)}>NO n</span> : '') : <span title={`n of ${r.nFrom}`}>{n(r.n)}</span>) },
        ]}
      />
      <Raw label="the evaluation record, verbatim" value={d.eval} />
    </>
  );
}

function StageBody({ stage, d, busy, act }: { stage: string; d: OnboardingDetail; busy: boolean; act: Act }) {
  switch (stage) {
    case 'resolve':
      return <Resolve d={d} />;
    case 'clarify':
      return <Licence d={d} busy={busy} act={act} />;
    case 'index':
      return <Index d={d} />;
    case 'vocab':
      return <Vocab d={d} busy={busy} act={act} />;
    case 'examples':
      return <Examples d={d} />;
    case 'knn':
      return <Knn d={d} />;
    case 'sources':
      return <Sources d={d} busy={busy} act={act} />;
    case 'skills':
      return <Skills d={d} />;
    case 'retire':
      return <Retire d={d} />;
    case 'rebuild':
      return <p {...stylex.props(text.bodySm, p.why)}>a JOIN: the skill document index and the example kNN index rebuild when the stack is idle</p>;
    case 'evaluate':
      return <Evaluation d={d} />;
    default:
      return null;
  }
}

function StagePanel({ cell, d, busy, act }: { cell: RailCell; d: OnboardingDetail; busy: boolean; act: Act }) {
  const st = cell.stage;
  const j = cell.job;
  const reached = stageRecord(d, st) !== null || cell.state !== 'pending';
  const t = RAIL_TONE[cell.state];
  const at = d.stage === st;
  return (
    <Panel
      kanji={STAGE_KANJI[st] ?? '段'}
      title={st.toUpperCase()}
      tag={cell.state === 'held' ? 'NEEDS YOU' : cell.state.toUpperCase()}
      tagTone={t ?? 'muted'}
      edge={at ? (t ?? undefined) : undefined}
      flag={cell.join ? 'JOIN' : j ? j.lane ?? undefined : undefined}
      sub={j ? `${j.queue} ${j.id} · ${j.state}${j.attempts !== null ? ` · try ${j.attempts}/${j.max_attempts ?? '—'}` : ''}${j.finished ? ` · ${ago(Date.now() / 1000 - j.finished)} ago` : ''}` : undefined}
    >
      {j?.progress && j.state !== 'done' ? <p {...stylex.props(text.bodySm, p.why)}>{j.progress}</p> : null}
      {j?.error ? <p {...stylex.props(text.bodySm, j.state === 'errored' ? p.err : p.why)}>{j.error}</p> : null}
      {reached ? <StageBody stage={st} d={d} busy={busy} act={act} /> : <StateView kind="inert" title="not reached yet" />}
      {j?.result !== undefined && j.result !== null ? <Raw label="the job's result" value={j.result} /> : null}
      <ReviewBox d={d} stage={st} busy={busy} act={act} />
    </Panel>
  );
}

// ---------------------------------------------------------------------------
function Header({ d, stale, busy, act }: { d: OnboardingDetail; stale: boolean; busy: boolean; act: Act }) {
  const now = Date.now() / 1000;
  const waiting = waitingLine(d);
  const notes = d.notes ?? {};
  const rerun = (job: string) => void act(DATASET_POST.rerun, { job }, `job ${job} re-queued`);
  return (
    <Panel kanji="技" title={packageLabel(d)} tag={d.state.toUpperCase()} tagTone={onboardingTone(d.state)} edge={onboardingTone(d.state)} flag={d.id} stale={stale} sub={`at ${d.stage}${d.next_stage ? ` · next ${d.next_stage}` : ''} · updated ${ago(now - d.updated)} ago`}>
      {d.prompt ? <p {...stylex.props(text.bodySm, p.quote)}>{d.prompt}</p> : null}
      <KV
        rows={[
          ...(notes.links?.length ? ([['links', notes.links.join(' ')]] as [string, ReactNode][]) : []),
          ...(notes.aliases?.length ? ([['aliases', notes.aliases.join(', ')]] as [string, ReactNode][]) : []),
          ['replaces', notes.replaces ?? 'auto (by major)'],
          ...(notes.author ? ([['by', notes.author]] as [string, ReactNode][]) : []),
          ...(d.group ? ([['group', d.group]] as [string, ReactNode][]) : []),
        ]}
      />
      <div {...stylex.props(layout.grid4)}>
        <Stat label="jobs done" value={`${n(d.jobs.done ?? 0)}/${n(d.jobs.total ?? 0)}`} />
        <Stat label="running · queued" value={`${n(d.jobs.running ?? 0)} · ${n(d.jobs.queued ?? 0)}`} />
        <Stat label="errored" value={n(d.jobs.errored ?? 0)} tone={d.jobs.errored ? 'crimson' : undefined} />
        <Stat label="skills" value={n(d.skills.length)} sub={d.skill_counts.armed ? `${d.skill_counts.armed} armed` : undefined} />
      </div>
      <OnboardingRail d={d} now={now} />
      {waiting ? <p {...stylex.props(text.bodySm, p.why)}>WAITING · {waiting}</p> : null}
      <Blockers blockers={d.blockers} busy={busy} onRerun={rerun} />
      <ErroredJobs rows={d.job_rows} blockers={d.blockers} busy={busy} onRerun={rerun} />
      {d.stage === 'clarify' ? <ClarifyForms d={d} busy={busy} act={act} /> : null}
      {d.warnings.map((w, i) => (
        <Row key={i} bad={w.kind === 'licence'}>
          <span {...stylex.props(p.why)}>
            {w.kind}: {w.what}
          </span>
        </Row>
      ))}
    </Panel>
  );
}

export function OnboardingScreen({ id }: { id: string }) {
  const [every, setEvery] = useState(5000);
  const d = usePoll<{ onboarding: OnboardingDetail }>(onboardingPath(id), every);
  const { busy, msg, act } = useAct(d.reload);
  const x = d.data?.onboarding ?? null;
  useEffect(() => setEvery(x?.stage === 'complete' ? 30000 : 5000), [x?.stage]);
  const nav = <SubNav current={null} extra={<NavLink to={href.skillsView('create')} {...stylex.props(text.labelXs, p.link)}>← CREATE · WATCHING</NavLink>} />;
  if (!x) {
    return (
      <div {...stylex.props(s.page)}>
        {nav}
        <PollState path={onboardingPath(id)} failure={d.failure} />
      </div>
    );
  }
  const cells = railCells(x, Date.now() / 1000).filter((c) => c.stage !== 'submitted' && c.stage !== 'complete');
  return (
    <div {...stylex.props(s.page)}>
      {nav}
      <Header d={x} stale={d.stale} busy={busy} act={act} />
      <Note msg={msg} />
      <div {...stylex.props(s.grid)}>
        {cells.map((c) => (
          <StagePanel key={c.stage} cell={c} d={x} busy={busy} act={act} />
        ))}
      </div>
      <Raw label="the onboarding record, verbatim (GET /dash/api/skill-factory/onboarding/<id>)" value={x} />
    </div>
  );
}
