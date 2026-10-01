// 技 SKILLS — the skill factory (NAEDOKO's skills side; docs/SKILL-FACTORY.md).
// Four views over the one pipeline:
//
//   LIBRARY     every skill, faceted by the taxonomy the server sends, with
//               each skill's injected size against the caps it sends; the
//               served skills by area and the held packages (folded in from
//               the retired NEBARI screen, 2026-09-30)
//   CREATE      paste text, give URL(s), or hand over a frontier SKILL.md;
//               watch each stage answer, and why it stopped
//   SELECTIONS  the injector's jjava decisions per model (/dash/api/jjava),
//               what the last requests were given (x_yamadori.skills), and
//               the durable selection log
//   PROMPTS     the versioned templates the model stages run
//
// A skill arms with no review; a person reviews here whenever they like.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useMemo, useState } from 'react';
import {
  facetCounts,
  facetsOf,
  filterSkills,
  groupDecisions,
  isFrontierUrl,
  isSkillMd,
  overallFallbackRate,
  PROMPTS_PATH,
  RECENT_PATH,
  SELECTIONS_PATH,
  SKILL_POST,
  SKILLS_PATH,
  skillPath,
  sortSkills,
  splitRemedy,
  statusTone,
  submitBody,
  toggle,
  URL_RE,
  canRerun,
  type CreateMode,
  type Facet,
  type Filters,
  type PromptMeta,
  type RecentRequests,
  type Selection,
  type SkillDetail,
  type SkillsOverview,
  type SkillSummary,
  type SortKey,
} from '../api/skills';
import { linksOf, ONBOARDING_PATH, onboardingBody, onboardingTone, packageLabel, splitList, type OnboardingSummary, type PromptFields, type ReplacesChoice } from '../api/onboarding';
import { LIBRARY_PATH, type Library as LibraryData } from '../api/library';
import { JJAVA_PATH, type Jjava } from '../api/stats';
import { usePoll } from '../api/usePoll';
import { ago, clock, n } from '../format';
import { href, type SkillsView } from '../router';
import { colors, space } from '../tokens/tokens.stylex';
import { NavLink } from '../ui/NavLink';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Row, Stat } from '../ui/primitives';
import { InjectorPanel } from '../ui/StatsParts';
import { StrataPanel } from '../ui/Strata';
import { useShared } from '../api/data';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';
import { AreasPanel, PackagesPanel } from './skills/library';
import { OnboardingCard } from './skills/onboarding';
import { Btn, Chips, Field, Note, ProhibitionChip, s as p, send, SizeBar, StageRail, SubNav, type Msg } from './skills/parts';

const s = stylex.create({
  page: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  lib: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'minmax(240px, 1fr) minmax(0, 3fr)' },
    alignItems: 'start',
  },
  two: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'minmax(0, 3fr) minmax(0, 2fr)' },
    alignItems: 'start',
  },
  col: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  facet: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: space.spaceXs,
    backgroundColor: colors.surfaceContainerLowest,
    color: colors.onSurfaceVariant,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 60%, transparent)`,
    paddingInline: space.spaceXs,
    paddingBlock: '1px',
    cursor: 'pointer',
    maxWidth: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    textTransform: 'none',
  },
  facetOn: {
    color: colors.primaryContainer,
    borderColor: colors.primaryContainer,
    backgroundColor: `color-mix(in srgb, ${colors.primaryContainer} 12%, transparent)`,
  },
  facetZero: { opacity: 0.45 },
  count: { color: colors.outline, fontVariantNumeric: 'tabular-nums' },
  cellTitle: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: '12rem', maxWidth: '28rem' },
  desc: { color: colors.onSurfaceVariant, overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', textTransform: 'none' },
  sizeCell: { minWidth: '7rem' },
  mode: { display: 'flex', flexWrap: 'wrap', gap: space.spaceSm },
  radio: { display: 'inline-flex', alignItems: 'center', gap: space.spaceXs, cursor: 'pointer', color: colors.onSurface },
  fields: { display: 'flex', flexWrap: 'wrap', gap: space.spaceSm },
  card: {
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceSm,
    padding: space.spaceSm,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 35%, transparent)`,
    minWidth: 0,
  },
});

// ---------------------------------------------------------------------------
// Per-viewer conveniences: the library's filters and the skills this browser
// submitted and is watching. Losing either loses nothing server-side.
// ---------------------------------------------------------------------------
const FILTER_KEY = 'yamadori_skills_filters';
const WATCH_KEY = 'yamadori_skills_watch';
const ONBOARD_KEY = 'yamadori_onboarding_watch';

function readJson<T>(key: string, fallback: T, ok: (v: unknown) => boolean): T {
  try {
    const v = JSON.parse(localStorage.getItem(key) ?? 'null');
    return ok(v) ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}
function writeJson(key: string, v: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* storage blocked: the state lasts until the page is reloaded */
  }
}

const isFilters = (v: unknown) => !!v && typeof v === 'object' && !Array.isArray(v);

// ---------------------------------------------------------------------------
// LIBRARY
// ---------------------------------------------------------------------------
const FACET_SHOW = 8;
const PAGE = 100;

function FacetGroup({ facet, counts, selected, onToggle }: { facet: Facet; counts: [string, number][]; selected: string[]; onToggle: (v: string) => void }) {
  const [all, setAll] = useState(false);
  if (!counts.length) return null;
  const shown = all ? counts : counts.slice(0, FACET_SHOW);
  return (
    <div {...stylex.props(layout.stack)}>
      <Label>
        {facet.label}
        {selected.length ? ` · ${selected.length} ON` : ''}
      </Label>
      <div {...stylex.props(layout.rowWrap)}>
        {shown.map(([v, c]) => {
          const on = selected.includes(v);
          return (
            <button key={v} type="button" aria-pressed={on} onClick={() => onToggle(v)} title={facet.names[v] && facet.names[v] !== v ? `${facet.names[v]} (${v})` : v} {...stylex.props(text.labelXs, s.facet, on && s.facetOn, !c && !on && s.facetZero)}>
              <span>{facet.names[v] ?? v}</span>
              <span {...stylex.props(s.count)}>{n(c)}</span>
            </button>
          );
        })}
        {counts.length > FACET_SHOW ? (
          <button type="button" onClick={() => setAll((a) => !a)} {...stylex.props(text.labelXs, s.facet)}>
            {all ? 'fewer' : `+${counts.length - FACET_SHOW} more`}
          </button>
        ) : null}
      </div>
    </div>
  );
}

/** The library table. Presentational, so it renders in a test. */
export function SkillTable({ skills, o }: { skills: SkillSummary[]; o: Pick<SkillsOverview, 'limits' | 'taxonomy'> }) {
  if (!skills.length) return <StateView kind="empty" title="no skill matches" detail="clear a filter or the search" />;
  const l = o.limits;
  const names = Object.fromEntries(facetsOf(o).flatMap((f) => Object.entries(f.names)));
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
            <span {...stylex.props(s.cellTitle)}>
              <NavLink to={href.skill(x.id)} {...stylex.props(text.bodySm, p.link)}>
                {x.title || x.name}
              </NavLink>
              {x.description ? <span {...stylex.props(text.labelXs, s.desc)}>{x.description}</span> : null}
            </span>
          ),
        },
        { key: 's', head: 'state', cell: (x) => <Chip tone={statusTone(x.status)}>{x.status}</Chip> },
        {
          key: 'c',
          head: 'filed under',
          cell: (x) => <Chips values={[...(x.category?.language ?? []), ...(x.category?.framework ?? []), ...(x.category?.phase ?? [])]} names={names} empty={x.category?.artifact?.map((a) => names[a] ?? a).join(', ') || '—'} />,
        },
        {
          key: 'z',
          head: 'size',
          cell: (x) => (
            <span {...stylex.props(s.sizeCell)}>
              <SizeBar size={x.size} limits={l} compact />
            </span>
          ),
        },
        { key: 'i', head: 'items', num: true, cell: (x) => (x.size ? `${x.size.items}/${l.max_items}` : '—') },
        { key: 'p', head: 'do not', num: true, cell: (x) => <ProhibitionChip n={x.size?.prohibitions} max={l.max_prohibitions} /> },
        {
          key: 'a',
          head: 'tests',
          num: true,
          cell: (x) =>
            x.activation ? (
              <Chip tone={x.activation.passed ? 'moss' : 'crimson'} title={`${x.activation.n ?? 0} case(s)`}>
                {x.activation.score === null ? '—' : `${Math.round(x.activation.score * 100)}%`}
              </Chip>
            ) : (
              '—'
            ),
        },
      ]}
    />
  );
}

function Library({ o, stale }: { o: SkillsOverview; stale: boolean }) {
  const facets = useMemo(() => facetsOf(o), [o]);
  const [filters, setFilters] = useState<Filters>(() => readJson<Filters>(FILTER_KEY, {}, isFilters));
  const [q, setQ] = useState('');
  const [sort, setSort] = useState<SortKey>('title');
  const [limit, setLimit] = useState(PAGE);
  const [open] = useState(() => (typeof window === 'undefined' ? true : window.matchMedia('(min-width: 1100px)').matches));
  useEffect(() => writeJson(FILTER_KEY, filters), [filters]);
  const hits = useMemo(() => sortSkills(filterSkills(o.skills, facets, filters, q), sort), [o.skills, facets, filters, q, sort]);
  const active = Object.values(filters).reduce((a, v) => a + v.length, 0);
  const l = o.limits;
  const sized = o.skills.filter((x) => x.size);
  const overAim = sized.filter((x) => x.size!.tokens > l.skill_tokens_aim[1]).length;
  const overHard = sized.filter((x) => x.size!.tokens > l.skill_tokens_hard).length;
  const atProh = sized.filter((x) => x.size!.prohibitions >= l.max_prohibitions).length;
  const failing = o.skills.filter((x) => x.activation && !x.activation.passed).length;
  const lib = usePoll<LibraryData>(LIBRARY_PATH, 60000);
  const { vitals } = useShared();
  return (
    <div {...stylex.props(s.lib)}>
      <Panel kanji="類" title="FACETS" tag={active ? `${active} ON` : 'ALL'} tagTone={active ? 'cyan' : 'muted'}>
        <details open={open}>
          <summary {...stylex.props(text.labelXs, p.summary)}>filter by taxonomy, state, provenance, licence</summary>
          <div {...stylex.props(layout.stackSm)} style={{ marginTop: 8 }}>
            {facets.map((f) => (
              <FacetGroup key={f.key} facet={f} counts={facetCounts(o.skills, facets, filters, q, f)} selected={filters[f.key] ?? []} onToggle={(v) => { setFilters(toggle(filters, f.key, v)); setLimit(PAGE); }} />
            ))}
            {active ? <Btn onClick={() => setFilters({})}>CLEAR FILTERS</Btn> : null}
          </div>
        </details>
      </Panel>
      <div {...stylex.props(s.col)}>
        <Panel
          kanji="技"
          title="LIBRARY"
          tag={`${n(hits.length)} OF ${n(o.skills.length)}`}
          tagTone="cyan"
          stale={stale}
          sub={Object.entries(o.counts).filter(([, v]) => v).map(([k, v]) => `${k} ${n(v)}`).join(' · ') || 'no skills'}
        >
          <div {...stylex.props(layout.grid4)}>
            <Stat label="over the aim" value={sized.length ? n(overAim) : '—'} sub={`> ${l.skill_tokens_aim[1]} tok`} tone={overAim ? 'cyan' : undefined} />
            <Stat label="over the hard cap" value={sized.length ? n(overHard) : '—'} sub={`> ${l.skill_tokens_hard} tok`} tone={overHard ? 'crimson' : undefined} />
            <Stat label="at the do-not cap" value={sized.length ? n(atProh) : '—'} sub={`${l.max_prohibitions} per skill`} />
            <Stat label="tests failing" value={n(failing)} tone={failing ? 'crimson' : undefined} />
          </div>
          {!sized.length && o.skills.length ? <p {...stylex.props(text.bodySm, p.why)}>sizes: this server's /dash/api/skills carries no `size` per skill</p> : null}
          <div {...stylex.props(s.fields)}>
            <Field label="search">
              <input type="search" aria-label="search skills" placeholder="name, title, description, tag, topic, trigger, id" value={q} onChange={(e) => { setQ(e.target.value); setLimit(PAGE); }} {...stylex.props(text.bodySm, p.input)} />
            </Field>
            <Field label="sort">
              <select aria-label="sort skills" value={sort} onChange={(e) => setSort(e.target.value as SortKey)} {...stylex.props(text.bodySm, p.input)}>
                <option value="title">title</option>
                <option value="updated">last updated</option>
                <option value="tokens">size, largest first</option>
                <option value="score">tests, weakest first</option>
              </select>
            </Field>
          </div>
          <SkillTable skills={hits.slice(0, limit)} o={o} />
          {hits.length > limit ? <Btn onClick={() => setLimit((x) => x + PAGE)}>{`SHOW ${Math.min(PAGE, hits.length - limit)} MORE OF ${n(hits.length - limit)}`}</Btn> : null}
          <Label>CAPS · {l.label}</Label>
        </Panel>
        <AreasPanel poll={lib} />
        <PackagesPanel poll={lib} />
        {/* TIER STRATA: the code-search result tiers over the held package
            indexes, 24 h (/dash/api/vitals strata; moved from NEBARI). */}
        {vitals.data ? <StrataPanel strata={vitals.data.strata} present={'strata' in vitals.data} /> : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CREATE
// ---------------------------------------------------------------------------
/** The form's modes: the three skill submissions, and a package onboarding. */
type FormMode = CreateMode | 'prompt';

const MODES: { mode: FormMode; label: string; what: string }[] = [
  { mode: 'text', label: 'PASTE TEXT', what: 'distilled into ONE atomic skill' },
  { mode: 'urls', label: 'URL(S)', what: 'fetched; a SKILL.md URL or a GitHub tree/ folder is decomposed, anything else distilled' },
  { mode: 'frontier', label: 'FRONTIER SKILL.md', what: 'decomposed into atomic skills; each child is classified, tested and validated on its own' },
  { mode: 'prompt', label: 'PROMPT + LINKS', what: 'a package onboarding: the links are resolved to package@version and a commit, then indexed, its vocabulary built, its examples and skill sources read; one onboarding per package, each watched below' },
];

const REPLACES: { value: ReplacesChoice; label: string; what: string }[] = [
  { value: 'auto', label: 'AUTO (BY MAJOR)', what: 'the rule decides when a version is already held: the same major replaces it, a new major sits alongside' },
  { value: 'replace', label: 'REPLACE', what: 'the held version is retired once the new one arms' },
  { value: 'alongside', label: 'ALONGSIDE', what: 'both versions stay held and served' },
];

/** A prompt with links: the POST /dash/api/skill onboarding form. */
function PromptForm({ onMade }: { onMade: (id: string) => void }) {
  const [f, setF] = useState<PromptFields>({ prompt: '', links: '', aliases: '', replaces: 'auto' });
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const found = linksOf(f.prompt, splitList(f.links, true));
  const submit = async () => {
    const b = onboardingBody(f);
    if ('error' in b) {
      setMsg({ tone: 'err', text: b.error });
      return;
    }
    setBusy(true);
    const r = await send<{ onboarding?: { id: string; stage: string } }>(SKILL_POST.create, b.body, 'submitted: resolve is queued');
    setBusy(false);
    setMsg(r.msg);
    const id = r.data?.onboarding?.id;
    if (id) {
      onMade(id);
      setF((x) => ({ ...x, prompt: '', links: '', aliases: '' }));
    }
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
      {...stylex.props(layout.stackSm)}
    >
      <textarea
        aria-label="the prompt"
        placeholder="Add koota 0.6.6, the ECS we use with r3f: https://github.com/pmndrs/koota"
        value={f.prompt}
        onChange={(e) => setF({ ...f, prompt: e.target.value })}
        {...stylex.props(text.bodySm, p.input, p.area)}
      />
      <div {...stylex.props(s.fields)}>
        <Field label="links (optional; beside those in the prompt)">
          <input aria-label="links" value={f.links} onChange={(e) => setF({ ...f, links: e.target.value })} placeholder="https://www.npmjs.com/package/… https://github.com/…" {...stylex.props(text.bodySm, p.input)} />
        </Field>
        <Field label="aliases (optional)">
          <input aria-label="aliases" value={f.aliases} onChange={(e) => setF({ ...f, aliases: e.target.value })} placeholder="r3f, fiber   or   r3f=@react-three/fiber" {...stylex.props(text.bodySm, p.input)} />
        </Field>
      </div>
      <p {...stylex.props(text.bodySm, p.why)}>
        aliases: the words a person uses for the package, comma separated -- the detector's rule, so typed here, never guessed. When the prompt names several packages, bind each as alias=package; an unbound alias applies only when the prompt names one package.
      </p>
      <div role="radiogroup" aria-label="a version already held" {...stylex.props(s.mode)}>
        <Label>A VERSION ALREADY HELD</Label>
        {REPLACES.map((r) => (
          <label key={r.value} title={r.what} {...stylex.props(text.labelMd, s.radio)}>
            <input type="radio" name="onboard-replaces" checked={f.replaces === r.value} onChange={() => setF({ ...f, replaces: r.value })} />
            {r.label}
          </label>
        ))}
      </div>
      <p {...stylex.props(text.bodySm, p.why)}>{REPLACES.find((r) => r.value === f.replaces)?.what}</p>
      <Label>{found.length ? `LINKS · ${found.length} · ${found.join(' · ')}` : 'LINKS · none yet: give at least one, in the prompt or the field'}</Label>
      <div {...stylex.props(layout.rowWrap)}>
        <Btn type="submit" disabled={busy || !f.prompt.trim() || !found.length}>
          ONBOARD
        </Btn>
      </div>
      <Note msg={msg} />
    </form>
  );
}

/** The path a submission will walk, from the overview's own `paths`. */
function pathFor(mode: CreateMode, input: string, o: SkillsOverview): { kind: string; path: string[] } {
  if (mode === 'urls') {
    const urls = input.split(/\s+/).filter((u) => URL_RE.test(u));
    const kind = urls.length && urls.every(isFrontierUrl) ? 'frontier' : 'url';
    return { kind, path: o.paths[kind] ?? [] };
  }
  const kind = mode === 'frontier' || isSkillMd(input) ? 'frontier_text' : 'text';
  return { kind, path: o.paths[kind] ?? [] };
}

function promptsByStage(o: Pick<SkillsOverview, 'prompts'>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const pr of o.prompts ?? []) out[pr.stage] = out[pr.stage] ? `${out[pr.stage]}, ${pr.version}` : pr.version;
  return out;
}

/** One submitted skill, polled while it is in the pipeline. */
function PipelineCard({ id, o, onDrop }: { id: string; o: SkillsOverview; onDrop: () => void }) {
  const [every, setEvery] = useState(3000);
  const d = usePoll<{ skill: SkillDetail }>(skillPath(id), every);
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const [spdx, setSpdx] = useState('');
  const [quote, setQuote] = useState('');
  const x = d.data?.skill;
  useEffect(() => {
    if (x) setEvery(x.status === 'pipeline' ? 3000 : 20000);
  }, [x?.status]);
  if (!x) {
    return (
      <div {...stylex.props(s.card)}>
        <PollState path={skillPath(id)} failure={d.failure} />
        {d.failure ? <Btn onClick={onDrop}>STOP WATCHING</Btn> : null}
      </div>
    );
  }
  const v = x.versions[0];
  const why = splitRemedy(v?.reason ?? x.reason);
  const act = async (path: string, body: unknown, ok: string) => {
    setBusy(true);
    const r = await send(path, body, ok);
    setBusy(false);
    setMsg(r.msg);
    if (r.data) d.reload();
  };
  const licenceStop = v?.state === 'failed' && v.stage === 'licence';
  const jobErrors = x.jobs.filter((j) => j.error).slice(0, 3);
  return (
    <div {...stylex.props(s.card)} aria-label={`submission ${x.name}`}>
      <div {...stylex.props(layout.between)}>
        <NavLink to={href.skill(x.id)} {...stylex.props(text.labelMd, p.link)}>
          {x.title || x.name}
        </NavLink>
        <span {...stylex.props(layout.rowWrap)}>
          <Chip tone={statusTone(x.status)}>{x.status}</Chip>
          <Chip tone="muted">{x.source_kind}</Chip>
        </span>
      </div>
      {v ? <StageRail stages={o.stages} version={v} jobs={x.jobs} prompts={promptsByStage(o)} /> : <StateView kind="inert" title="no version yet" />}
      {why.situation && v && v.state !== 'running' && v.state !== 'armed' ? (
        <div {...stylex.props(layout.stack)}>
          <p {...stylex.props(text.bodySm, v.state === 'decomposed' ? p.why : p.err)}>
            {v.state.toUpperCase()} AT {v.stage}: {why.situation}
          </p>
          {why.remedy ? <p {...stylex.props(text.bodySm, p.why)}>REMEDY {why.remedy}{why.retryable ? ` · retryable: ${why.retryable}` : ''}</p> : null}
        </div>
      ) : null}
      {jobErrors.map((j) => (
        <Row key={j.id} bad>
          <span {...stylex.props(p.err)}>
            {j.stage ?? j.queue}: {j.error}
          </span>
        </Row>
      ))}
      {x.children.length ? (
        <div {...stylex.props(layout.stack)}>
          <Label>ATOMIC SKILLS FROM IT · {x.children.length}</Label>
          {x.children.map((c) => (
            <Row key={c.id}>
              <NavLink to={href.skill(c.id)} {...stylex.props(p.link)}>
                {c.name}
              </NavLink>
              <Chip tone={statusTone(c.status)}>{c.status}</Chip>
            </Row>
          ))}
        </div>
      ) : null}
      {licenceStop ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void act(SKILL_POST.licence, { id: x.id, licence: spdx, quote }, 'licence recorded').then(() => act(SKILL_POST.rerun, { id: x.id, stage: 'licence' }, 'licence stage re-run'));
          }}
          {...stylex.props(layout.stackSm)}
        >
          <div {...stylex.props(s.fields)}>
            <Field label="licence (SPDX)">
              <input value={spdx} onChange={(e) => setSpdx(e.target.value)} {...stylex.props(text.bodySm, p.input)} />
            </Field>
            <Field label="the text that grants it, verbatim">
              <input value={quote} onChange={(e) => setQuote(e.target.value)} {...stylex.props(text.bodySm, p.input)} />
            </Field>
          </div>
          <Btn type="submit" disabled={busy || !spdx.trim() || !quote.trim()}>
            RECORD LICENCE · RE-RUN
          </Btn>
        </form>
      ) : null}
      <div {...stylex.props(layout.rowWrap)}>
        {v && canRerun(v) && v.state !== 'running' ? (
          <Btn disabled={busy} onClick={() => void act(SKILL_POST.rerun, { id: x.id, stage: v.stage }, `${v.stage} re-run`)}>
            {`RE-RUN ${v.stage}`}
          </Btn>
        ) : null}
        <Btn onClick={onDrop}>STOP WATCHING</Btn>
        <span {...stylex.props(text.labelXs, text.outline)}>updated {ago(Date.now() / 1000 - x.updated)} ago</span>
      </div>
      <Note msg={msg} />
    </div>
  );
}

function Create({ o, onSubmitted }: { o: SkillsOverview; onSubmitted: () => void }) {
  const [mode, setMode] = useState<FormMode>('text');
  const [input, setInput] = useState('');
  const [goal, setGoal] = useState('');
  const [name, setName] = useState('');
  const [watchHours, setWatchHours] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const [watch, setWatch] = useState<string[]>(() => readJson<string[]>(WATCH_KEY, [], Array.isArray).filter((x) => typeof x === 'string').slice(0, 12));
  useEffect(() => writeJson(WATCH_KEY, watch), [watch]);
  const [boards, setBoards] = useState<string[]>(() => readJson<string[]>(ONBOARD_KEY, [], Array.isArray).filter((x) => typeof x === 'string').slice(0, 24));
  useEffect(() => writeJson(ONBOARD_KEY, boards), [boards]);
  const listing = usePoll<{ onboardings: OnboardingSummary[] }>(ONBOARDING_PATH, 10000);
  const all = listing.data?.onboardings ?? [];
  // Every onboarding still walking its stages, and the ones this browser submitted.
  // Each card polls its own detail, so the list is capped; the ONBOARDINGS table lists them all.
  const shownBoards = [...boards, ...all.filter((x) => x.stage !== 'complete' && !boards.includes(x.id)).map((x) => x.id)].slice(0, 12);
  const plan = pathFor(mode === 'prompt' ? 'text' : mode, input, o);
  const byStage = promptsByStage(o);
  const detected = mode === 'text' && isSkillMd(input);
  const inFlight = o.skills.filter((x) => x.status !== 'armed' && x.status !== 'archived' && !watch.includes(x.id));
  const submit = async () => {
    if (mode === 'prompt') return;
    const b = submitBody({ mode, input, goal, name, watchHours });
    if ('error' in b) {
      setMsg({ tone: 'err', text: b.error });
      return;
    }
    setBusy(true);
    const r = await send<{ skill?: SkillSummary; skills?: SkillSummary[] }>(SKILL_POST.create, b.body, 'submitted: the first stage is queued');
    setBusy(false);
    setMsg(r.msg);
    const made = r.data ? (r.data.skills ?? (r.data.skill ? [r.data.skill] : [])) : [];
    if (made.length) {
      setWatch((w) => [...made.map((m) => m.id), ...w.filter((id) => !made.some((m) => m.id === id))].slice(0, 12));
      setInput('');
      onSubmitted();
    }
  };
  return (
    <div {...stylex.props(s.two)}>
      <div {...stylex.props(s.col)}>
        <Panel kanji="加" title="SUBMIT" tag="ARMS WITHOUT REVIEW" tagTone="cyan">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
            {...stylex.props(layout.stackSm)}
          >
            <div role="radiogroup" aria-label="what you are giving" {...stylex.props(s.mode)}>
              {MODES.map((m) => (
                <label key={m.mode} {...stylex.props(text.labelMd, s.radio)}>
                  <input type="radio" name="skill-mode" checked={mode === m.mode} onChange={() => setMode(m.mode)} />
                  {m.label}
                </label>
              ))}
            </div>
            <p {...stylex.props(text.bodySm, p.why)}>{MODES.find((m) => m.mode === mode)?.what}</p>
            {mode === 'prompt' ? null : (
            <>
            <textarea
              aria-label={mode === 'urls' ? 'one URL per line' : mode === 'frontier' ? 'the SKILL.md' : 'the source text'}
              placeholder={mode === 'urls' ? 'https://… one per line (a SKILL.md, a GitHub tree/ folder, a docs page)' : mode === 'frontier' ? '---\nname: …\ndescription: Use when …\n---\n# …' : 'the source: docs, notes, a post-mortem …'}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              {...stylex.props(text.bodySm, p.input, p.area, mode !== 'urls' && p.areaTall)}
            />
            {detected ? <p {...stylex.props(text.bodySm, p.ok)}>this is a SKILL.md: it walks the frontier path and is decomposed</p> : null}
            <div {...stylex.props(s.fields)}>
              <Field label="goal (steers distil / decompose)">
                <input value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="a skill for …" {...stylex.props(text.bodySm, p.input)} />
              </Field>
              <Field label="name (optional)">
                <input value={name} onChange={(e) => setName(e.target.value)} {...stylex.props(text.bodySm, p.input)} />
              </Field>
              {mode === 'urls' ? (
                <Field label="re-fetch every N hours (0 stops)">
                  <input inputMode="decimal" value={watchHours} onChange={(e) => setWatchHours(e.target.value)} placeholder="server default" {...stylex.props(text.bodySm, p.input)} />
                </Field>
              ) : null}
            </div>
            <Label>
              PATH · {plan.kind} · {plan.path.length ? plan.path.join(' → ') : 'the server sent no path for this kind'}
            </Label>
            <StageRail stages={o.stages} version={{ path: plan.path, stage: plan.path[0] ?? '', state: 'planned' }} prompts={byStage} />
            <div {...stylex.props(layout.rowWrap)}>
              <Btn type="submit" disabled={busy || !input.trim()}>
                SUBMIT
              </Btn>
            </div>
            <Note msg={msg} />
            </>
            )}
          </form>
          {mode === 'prompt' ? (
            <PromptForm
              onMade={(id) => {
                setBoards((b) => [id, ...b.filter((x) => x !== id)].slice(0, 24));
                listing.reload();
              }}
            />
          ) : null}
        </Panel>
        <Panel kanji="流" title="WATCHING" tag={`${watch.length + shownBoards.length}`} tagTone="cyan" sub="every onboarding still walking its stages, and what this browser submitted; every stage as it answers">
          {shownBoards.map((id) => (
            <OnboardingCard key={`ob-${id}`} id={id} onDrop={boards.includes(id) ? () => setBoards((b) => b.filter((x) => x !== id)) : undefined} />
          ))}
          {listing.failure && !listing.data ? <PollState path={ONBOARDING_PATH} failure={listing.failure} /> : null}
          {watch.map((id) => (
            <PipelineCard key={id} id={id} o={o} onDrop={() => setWatch((w) => w.filter((x) => x !== id))} />
          ))}
          {!watch.length && !shownBoards.length ? <StateView kind="empty" title="nothing submitted from this browser, and no onboarding in progress" /> : null}
        </Panel>
      </div>
      <div {...stylex.props(s.col)}>
        {all.length ? (
          <Panel kanji="包" title="ONBOARDINGS" tag={`${all.length}`} tagTone={all.some((x) => x.state === 'errored') ? 'crimson' : 'muted'} stale={listing.stale} sub="every package onboarding; each one's page has every stage's result">
            <Table
              tall
              rows={[...all].sort((a, b) => b.updated - a.updated)}
              rowKey={(x) => x.id}
              bad={(x) => x.state === 'errored'}
              columns={[
                {
                  key: 'p',
                  head: 'package',
                  cell: (x) => (
                    <NavLink to={href.onboarding(x.id)} {...stylex.props(p.link, p.asIs)}>
                      {packageLabel(x)}
                    </NavLink>
                  ),
                },
                { key: 's', head: 'state', cell: (x) => <Chip tone={onboardingTone(x.state)}>{x.state}</Chip> },
                { key: 'g', head: 'stage', cell: (x) => x.stage },
                { key: 'j', head: 'jobs', num: true, cell: (x) => `${x.jobs.done ?? 0}/${x.jobs.total ?? 0}${x.jobs.errored ? ` · ${x.jobs.errored} errored` : ''}` },
                { key: 'u', head: 'updated', cell: (x) => `${ago(Date.now() / 1000 - x.updated)} ago` },
              ]}
            />
          </Panel>
        ) : null}
        <Panel kanji="途" title="NOT ARMED" tag={`${inFlight.length}`} tagTone={inFlight.some((x) => x.status === 'failed' || x.status === 'quarantined') ? 'crimson' : 'muted'} sub="in the pipeline, stopped, disabled or decomposed">
          {inFlight.length ? (
            <Table
              tall
              rows={sortSkills(inFlight, 'updated')}
              rowKey={(x) => x.id}
              bad={(x) => x.status === 'failed' || x.status === 'quarantined'}
              columns={[
                {
                  key: 't',
                  head: 'skill',
                  cell: (x) => (
                    <NavLink to={href.skill(x.id)} {...stylex.props(p.link)}>
                      {x.title || x.name}
                    </NavLink>
                  ),
                },
                { key: 's', head: 'state', cell: (x) => <Chip tone={statusTone(x.status)}>{x.status}</Chip> },
                { key: 'r', head: 'why', cell: (x) => <span {...stylex.props(p.why)}>{splitRemedy(x.reason).situation || '—'}</span> },
                { key: 'w', head: '', cell: (x) => <Btn onClick={() => setWatch((w) => [x.id, ...w].slice(0, 12))}>WATCH</Btn> },
              ]}
            />
          ) : (
            <StateView kind="empty" title="every skill is armed or archived" />
          )}
        </Panel>
        <Panel kanji="型" title="PROMPTS" tag={`${o.prompts.length}`} tagTone="muted" sub="the versioned templates the model stages run; the model proposes, the code verifies">
          <Table
            rows={o.prompts}
            rowKey={(r) => r.name}
            columns={[
              { key: 'n', head: 'template', cell: (r) => r.version },
              { key: 's', head: 'stage', cell: (r) => r.stage },
              { key: 'c', head: 'chars', num: true, cell: (r) => n(r.chars) },
              { key: 'h', head: 'sha256', cell: (r) => <span {...stylex.props(text.labelXs, text.outline)}>{r.sha256}</span> },
            ]}
          />
          <NavLink to={href.skillsView('prompts')} {...stylex.props(text.labelXs, p.link)}>
            READ THE TEMPLATES →
          </NavLink>
        </Panel>
        {o.queue?.states ? (
          <Panel kanji="列" title="JOB QUEUE" tag={Object.keys(o.queue.paused ?? {}).length ? 'PAUSED' : 'RUNNING'} tagTone={Object.keys(o.queue.paused ?? {}).length ? 'crimson' : 'muted'}>
            <div {...stylex.props(layout.grid4)}>
              {Object.entries(o.queue.states).map(([k, v]) => (
                <Stat key={k} label={k} value={n(v)} tone={k === 'errored' && v ? 'crimson' : undefined} />
              ))}
            </div>
          </Panel>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SELECTIONS
// ---------------------------------------------------------------------------
function RecentPanel({ o }: { o: SkillsOverview }) {
  const r = usePoll<RecentRequests>(RECENT_PATH, 10000);
  const byId = useMemo(() => new Map(o.skills.map((x) => [x.id, x])), [o.skills]);
  if (!r.data) {
    return (
      <Panel kanji="今" title="RECENT REQUESTS" tag="x_yamadori.skills" tagTone="muted">
        <PollState path={RECENT_PATH} failure={r.failure} />
        {r.failure?.kind === 'http' && r.failure.status === 404 ? <p {...stylex.props(text.bodySm, p.why)}>the running server predates this endpoint</p> : null}
      </Panel>
    );
  }
  const l = r.data.limits;
  const reqs = r.data.requests;
  return (
    <Panel kanji="今" title="RECENT REQUESTS" tag={`${reqs.length} OF LAST ${r.data.keep}`} tagTone="cyan" stale={r.stale} sub="x_yamadori.skills of the proxy's last requests (memory; gone on a restart)">
      {reqs.length ? (
        reqs.map((q, i) => {
          const sk = q.skills;
          const tok = sk.tokens ?? 0;
          return (
            <div key={`${q.at}-${i}`} {...stylex.props(s.card)}>
              <div {...stylex.props(layout.between)}>
                <span {...stylex.props(text.labelMd, text.onSurface)}>
                  {clock(q.at)} · {sk.route_class ?? 'no route'}
                  {q.utility ? ' · side call' : ''}
                  {sk.replayed ? ' · replayed' : ''}
                </span>
                <Chip tone={sk.ids?.length ? 'moss' : 'muted'}>{sk.ids?.length ? `${sk.ids.length} injected` : sk.on === false ? 'off' : 'none'}</Chip>
              </div>
              {sk.why ? <p {...stylex.props(text.bodySm, p.why)}>{sk.why}</p> : null}
              {sk.ids?.length ? (
                <>
                  <span {...stylex.props(text.labelXs, text.outline, text.num)}>
                    {n(tok)} TOK THIS TURN · {n(sk.chars ?? 0)} CHARS · {sk.ids.length}/{l.max_skills_per_turn} SKILLS
                  </span>
                </>
              ) : null}
              {sk.matched.length ? (
                <Table
                  rows={sk.matched}
                  rowKey={(m) => m.id}
                  columns={[
                    {
                      key: 's',
                      head: 'skill',
                      cell: (m) => (
                        <NavLink to={href.skill(m.id)} {...stylex.props(p.link)}>
                          {m.title || byId.get(m.id)?.title || m.name || m.id}
                        </NavLink>
                      ),
                    },
                    { key: 'd', head: 'decided by', cell: (m) => `${m.decided_by ?? '—'} · ${m.strength ?? '—'}` },
                    { key: 'c', head: 'cos / conf', num: true, cell: (m) => `${m.cosine === null ? '—' : m.cosine.toFixed(2)} / ${m.confidence === null ? '—' : m.confidence.toFixed(2)}` },
                    { key: 'w', head: 'matched', cell: (m) => <Chips values={m.why ?? []} /> },
                  ]}
                />
              ) : null}
              {sk.dropped.length ? (
                <div {...stylex.props(layout.stack)}>
                  <Label>DROPPED</Label>
                  {sk.dropped.map((x, j) => (
                    <Row key={`${x.id}-${j}`}>
                      <NavLink to={href.skill(x.id)} {...stylex.props(p.link)}>
                        {byId.get(x.id)?.title || x.id}
                      </NavLink>
                      <span {...stylex.props(p.why)}>{x.why}</span>
                    </Row>
                  ))}
                </div>
              ) : null}
              {sk.candidates === undefined ? null : <span {...stylex.props(text.labelXs, text.outline)}>
                {sk.candidates} candidate(s) of {sk.armed ?? '—'} armed · cache {sk.cache ?? '—'} · embedding {sk.embedding ? (sk.embedding.ok ? 'answered' : `no: ${sk.embedding.why ?? ''}`) : '—'} · fallback {sk.fallback?.ran ? (sk.fallback.ok ? 'ran' : 'failed') : 'not run'}
              </span>}
            </div>
          );
        })
      ) : (
        <StateView kind="empty" title="no request since the proxy started" />
      )}
    </Panel>
  );
}

function SelectionLog({ o }: { o: SkillsOverview }) {
  const r = usePoll<{ selections: Selection[] }>(SELECTIONS_PATH, 15000);
  const byId = useMemo(() => new Map(o.skills.map((x) => [x.id, x])), [o.skills]);
  if (!r.data) return <PollState path={SELECTIONS_PATH} failure={r.failure} />;
  const decisions = groupDecisions(r.data.selections);
  return (
    <Panel kanji="録" title="SELECTION LOG" tag={`${decisions.length} DECISION(S)`} tagTone="cyan" stale={r.stale} sub="durable: one row per injected skill per decision; the request is a hash, never text">
      {decisions.length ? (
        <Table
          tall
          rows={decisions}
          rowKey={(d) => `${d.ts}|${d.request}`}
          columns={[
            { key: 't', head: 'when', cell: (d) => `${ago(Date.now() / 1000 - d.ts)} ago` },
            { key: 'c', head: 'class', cell: (d) => d.route_class ?? '—' },
            {
              key: 's',
              head: 'injected',
              cell: (d) => (
                <span {...stylex.props(layout.stack)}>
                  {d.rows.map((x) => (
                    <span key={x.skill}>
                      <NavLink to={href.skill(x.skill)} {...stylex.props(p.link)}>
                        {byId.get(x.skill)?.title || x.skill}
                      </NavLink>{' '}
                      <span {...stylex.props(text.labelXs, text.outline)}>
                        v{x.version ?? '—'} · {x.decided_by ?? '—'} · {x.strength ?? '—'}
                      </span>
                    </span>
                  ))}
                </span>
              ),
            },
            {
              key: 'k',
              head: 'est. tokens',
              num: true,
              cell: (d) => {
                const sizes = d.rows.map((x) => byId.get(x.skill)?.size?.tokens);
                if (sizes.some((z) => z === undefined)) return '—';
                const t = sizes.reduce<number>((a, z) => a + (z ?? 0), 0);
                return <span title="from each skill's served body now, not the text injected then">{n(t)}</span>;
              },
            },
            { key: 'r', head: 'request', cell: (d) => <span {...stylex.props(text.labelXs, text.outline)}>{d.request ?? '—'}</span> },
          ]}
        />
      ) : (
        <StateView kind="empty" title="no skill has been selected yet" />
      )}
    </Panel>
  );
}

function Learning({ o }: { o: SkillsOverview }) {
  const lr = o.learning;
  if (!lr) return null;
  const rate = overallFallbackRate(lr.rate);
  return (
    <Panel kanji="学" title="SELECTOR · FALLBACK RATE" tag={`${lr.rate.length} DAY(S)`} tagTone="muted" sub="no embedding thresholds: a cosine only ranks">
      <div {...stylex.props(layout.grid3)}>
        <Stat label="fallback rate" value={rate === null ? '—' : `${Math.round(rate * 100)}%`} />
        <Stat label="pending records" value={n(lr.pending)} />
        <Stat label="requests" value={n(lr.rate.reduce((a, d) => a + d.requests, 0))} />
      </div>
      <p {...stylex.props(text.bodySm, p.why)}>learning: {lr.idle.why}</p>
      {lr.rate.length ? (
        <Table
          rows={lr.rate}
          rowKey={(d) => d.day}
          columns={[
            { key: 'd', head: 'day', cell: (d) => d.day },
            { key: 'q', head: 'requests', num: true, cell: (d) => n(d.requests) },
            { key: 'c', head: 'with candidates', num: true, cell: (d) => n(d.with_candidates) },
            { key: 'i', head: 'injected', num: true, cell: (d) => n(d.injected) },
            { key: 'f', head: 'fallbacks', num: true, cell: (d) => n(d.fallbacks) },
            { key: 'r', head: 'rate', num: true, cell: (d) => (d.fallback_rate === null ? '—' : `${Math.round(d.fallback_rate * 100)}%`) },
          ]}
        />
      ) : null}
    </Panel>
  );
}

/** The skills injector's jjava decisions (/dash/api/jjava, the JJAVA page's
 *  same panel): what stage 1 offered, what jjava passed, injected or not. */
function InjectorStats() {
  const r = usePoll<Jjava>(JJAVA_PATH, 30000);
  if (!r.data && r.failure) return <PollState path={JJAVA_PATH} failure={r.failure} />;
  return <InjectorPanel data={r.data} stale={r.stale} />;
}

function Selections({ o }: { o: SkillsOverview }) {
  return (
    <div {...stylex.props(s.two)}>
      <div {...stylex.props(s.col)}>
        <InjectorStats />
        <RecentPanel o={o} />
      </div>
      <div {...stylex.props(s.col)}>
        <SelectionLog o={o} />
        <Learning o={o} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PROMPTS
// ---------------------------------------------------------------------------
function Prompts() {
  const r = usePoll<{ prompts: PromptMeta[] }>(PROMPTS_PATH, 0);
  if (!r.data) return <PollState path={PROMPTS_PATH} failure={r.failure} />;
  return (
    <div {...stylex.props(s.col)}>
      {r.data.prompts.map((pr) => (
        <Panel key={pr.name} kanji="型" title={pr.name.toUpperCase()} tag={pr.version} tagTone="cyan" sub={`stage ${pr.stage} · ${n(pr.chars)} chars · sha256 ${pr.sha256}`}>
          <details>
            <summary {...stylex.props(text.labelXs, p.summary)}>the template text</summary>
            <pre {...stylex.props(text.bodySm, p.pre)}>{pr.text ?? '—'}</pre>
          </details>
        </Panel>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
export function Skills({ view = 'library' }: { view?: SkillsView }) {
  const o = usePoll<SkillsOverview>(view === 'prompts' ? null : SKILLS_PATH, view === 'create' ? 10000 : 30000);
  const body = (() => {
    if (view === 'prompts') return <Prompts />;
    if (!o.data) return <PollState path={SKILLS_PATH} failure={o.failure} />;
    switch (view) {
      case 'create':
        return <Create o={o.data} onSubmitted={o.reload} />;
      case 'selections':
        return <Selections o={o.data} />;
      default:
        return <Library o={o.data} stale={o.stale} />;
    }
  })();
  return (
    <div {...stylex.props(s.page)}>
      <SubNav current={view} extra={o.data ? <span {...stylex.props(text.labelXs, text.outline)}>{`${n(o.data.skills.length)} SKILLS · ${n(o.data.counts.armed ?? 0)} ARMED`}</span> : null} />
      {body}
    </div>
  );
}
