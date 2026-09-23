// ADD DATASET: paste a URL or the text, and watch the pipeline answer for it.
//
// The worker fetches the source, then a model job (dataset.assist) reads it.
// Every clarifying field on the card wears where its value came from:
//
//   EVIDENCE   a verbatim quote the worker verified, with the quote and a link
//              to where it was found. The licence is ONLY ever filled this way.
//   PROPOSED   the model's reading of the source -- an inference, labelled.
//   OPERATOR   typed by a person.
//   NEEDS YOU  not established; the card says why it is asked and what the
//              assist already searched, so nobody repeats that search or
//              types a licence they did not check.
//
// Any field can be answered or overridden inline. Nothing here is a number
// the server did not send: job states and progress strings are printed
// verbatim, and there is no progress bar because nothing measures one.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useState } from 'react';
import {
  answerBody,
  badgeFor,
  displayValue,
  draftOf,
  jobLine,
  linkable,
  liveJobs,
  pollEvery,
  refusalText,
  submissionBody,
  waitingOn,
} from '../api/assist';
import { postJson } from '../api/client';
import { PATHS, useShared } from '../api/data';
import type { Dataset, DatasetWarning, FieldState, JobRow } from '../api/types';
import { usePoll } from '../api/usePoll';
import { href } from '../router';
import { colors, space } from '../tokens/tokens.stylex';
import { NavLink } from '../ui/NavLink';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Row } from '../ui/primitives';
import { StageTrack } from '../ui/StageTrack';
import { PollState, StateView } from '../ui/StateView';
import { text } from '../ui/text';

const s = stylex.create({
  input: {
    boxSizing: 'border-box',
    width: '100%',
    minWidth: 0,
    backgroundColor: colors.surfaceContainerLowest,
    color: colors.tertiaryContainer,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.outlineVariant,
    boxShadow: `inset 0 2px 6px color-mix(in srgb, ${colors.surfaceContainerLowest} 80%, black)`,
    paddingInline: space.spaceSm,
    paddingBlock: space.spaceXs,
    outline: { default: 'none', ':focus-visible': `1px solid ${colors.tertiaryContainer}` },
  },
  area: { minHeight: '6.5rem', resize: 'vertical' },
  select: { width: 'auto' },
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
  quiet: {
    color: colors.tertiaryContainer,
    borderColor: colors.outlineVariant,
    backgroundColor: { default: 'transparent', ':hover': `color-mix(in srgb, ${colors.tertiaryContainer} 10%, transparent)` },
  },
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceXs,
    padding: space.spaceSm,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
    minWidth: 0,
  },
  fieldNeeds: {
    borderColor: `color-mix(in srgb, ${colors.secondaryContainer} 55%, transparent)`,
    boxShadow: `inset 2px 0 0 ${colors.secondaryContainer}`,
  },
  fieldEvidence: { boxShadow: `inset 2px 0 0 ${colors.primaryContainer}` },
  fieldProposed: { boxShadow: `inset 2px 0 0 ${colors.tertiaryContainer}` },
  head: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: space.spaceXs, justifyContent: 'space-between' },
  value: { margin: 0, color: colors.onSurface, overflowWrap: 'anywhere' },
  unset: { margin: 0, color: colors.secondary },
  quote: {
    margin: 0,
    paddingInlineStart: space.spaceSm,
    borderInlineStartWidth: 2,
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: colors.primaryContainer,
    color: colors.primary,
    overflowWrap: 'anywhere',
    whiteSpace: 'pre-wrap',
  },
  note: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere', textTransform: 'none' },
  warn: { margin: 0, color: colors.secondary, overflowWrap: 'anywhere' },
  link: { color: colors.tertiaryContainer, textDecoration: 'none', overflowWrap: 'anywhere', ':hover': { textDecoration: 'underline' } },
  choices: { display: 'flex', flexWrap: 'wrap', gap: `${space.spaceXs} ${space.spaceMd}` },
  choice: { display: 'flex', alignItems: 'center', gap: space.spaceXs, color: colors.onSurfaceVariant, minHeight: '24px' },
  fields: { display: 'grid', gap: space.spaceXs, gridTemplateColumns: 'minmax(0, 1fr)' },
});

type Msg = { tone: 'busy' | 'ok' | 'err'; text: string } | null;

function MsgLine({ msg }: { msg: Msg }) {
  if (!msg) return null;
  return (
    <p role={msg.tone === 'err' ? 'alert' : 'status'} {...stylex.props(text.bodySm, msg.tone === 'err' ? s.warn : s.note)}>
      {msg.tone === 'err' ? '[ ! ] ' : ''}
      {msg.text}
    </p>
  );
}

// ---------------------------------------------------------------------------
// The submission form.
// ---------------------------------------------------------------------------
type Created = { ok: true; dataset: Dataset; missing: unknown[]; assist_jobs?: JobRow[] };

export function AddDatasetPanel({ onSubmitted, fill }: { onSubmitted: (id: string) => void; fill?: boolean }) {
  const [input, setInput] = useState('');
  const [kind, setKind] = useState('recipes');
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const body = submissionBody(input, kind);
  const shape = 'error' in body ? null : 'url' in body ? 'URL · THE WORKER FETCHES IT' : 'TEXT · READ AS PASTED';

  const submit = async () => {
    if ('error' in body) {
      setMsg({ tone: 'err', text: body.error });
      return;
    }
    setBusy(true);
    setMsg({ tone: 'busy', text: 'recording the submission' });
    const r = await postJson<Created>('/dash/api/dataset', body);
    setBusy(false);
    if (!r.ok) {
      setMsg({ tone: 'err', text: `not recorded: ${refusalText(r)}` });
      return;
    }
    const d = r.data.dataset;
    setMsg({
      tone: 'ok',
      text:
        `recorded as ${d.id}. ` +
        ('url' in body
          ? 'The fetch is queued on the net lane; the assist reads it once it lands.'
          : r.data.assist_jobs?.length
            ? 'The assist is queued on the gpu lane.'
            : 'Every field the assist may fill was already answered.'),
    });
    setInput('');
    onSubmitted(d.id);
  };

  return (
    <Panel
      kanji="植"
      title="ADD DATASET"
      tag="MODEL ASSISTED"
      tagTone="cyan"
      edge="cyan"
      fill={fill}
    >
      <form
        aria-label="add a dataset"
        {...stylex.props(layout.stackSm)}
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <label {...stylex.props(layout.stack)}>
          <Label>URL OR PASTED TEXT</Label>
          <textarea
            name="source"
            aria-label="URL or pasted text"
            rows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="https://… or the text to distil"
            {...stylex.props(text.bodySm, s.input, s.area)}
          />
        </label>
        <div {...stylex.props(layout.rowWrap)}>
          <label {...stylex.props(layout.rowWrap)}>
            <Label>KIND</Label>
            <select name="kind" value={kind} onChange={(e) => setKind(e.target.value)} {...stylex.props(text.bodySm, s.input, s.select)}>
              <option value="recipes">recipes</option>
              <option value="laya">laya training set</option>
            </select>
          </label>
          {shape && <Chip tone="muted">{shape}</Chip>}
          <span {...stylex.props(layout.grow)} />
          <button type="submit" disabled={busy || !input.trim()} {...stylex.props(text.labelMd, s.button)}>
            [ SUBMIT ]
          </button>
        </div>
        <MsgLine msg={msg} />
        <Label>The licence is filled only from a verbatim quote found in the source or its LICENSE file. Otherwise it is asked, never guessed.</Label>
      </form>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// One field, with its provenance badge and an inline answer / override.
// ---------------------------------------------------------------------------
type Answered = { ok: true; dataset: Dataset; missing: unknown[]; warnings: DatasetWarning[] };

export function ProvenanceBadge({ state }: { state: string }) {
  const b = badgeFor(state);
  return (
    <Chip tone={b.tone} title={b.title}>
      {b.label}
    </Chip>
  );
}

export function FieldRow({ datasetId, f, onSaved, canEdit = true }: { datasetId: string; f: FieldState; onSaved: (r: Answered) => void; canEdit?: boolean }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<string | string[]>(() => draftOf(f));
  const [msg, setMsg] = useState<Msg>(null);
  const needs = f.state === 'needs_you';

  const save = async () => {
    setMsg({ tone: 'busy', text: 'saving' });
    const r = await postJson<Answered>('/dash/api/dataset/answer', answerBody(datasetId, f, draft));
    if (!r.ok) {
      setMsg({ tone: 'err', text: `not saved: ${refusalText(r)}` });
      return;
    }
    setMsg(null);
    setEditing(false);
    onSaved(r.data);
  };

  const multi = f.input === 'multi';
  const picked = Array.isArray(draft) ? draft : [];
  return (
    <div
      data-field={f.field}
      data-state={f.state}
      {...stylex.props(s.field, needs && s.fieldNeeds, f.state === 'evidence' && s.fieldEvidence, f.state === 'proposed' && s.fieldProposed)}
    >
      <div {...stylex.props(s.head)}>
        <div {...stylex.props(layout.rowWrap)}>
          <span {...stylex.props(text.labelMd, text.primary)}>{f.label}</span>
          <ProvenanceBadge state={f.state} />
          {needs && f.severity === 'blocker' ? <Chip tone="crimson">BLOCKER</Chip> : null}
        </div>
        {canEdit && !editing ? (
          <button
            type="button"
            onClick={() => {
              setDraft(draftOf(f));
              setEditing(true);
            }}
            {...stylex.props(text.labelXs, s.button, s.quiet)}
          >
            {needs ? '[ ANSWER ]' : '[ OVERRIDE ]'}
          </button>
        ) : null}
      </div>

      <p {...stylex.props(text.bodySm, needs ? s.unset : s.value)}>{displayValue(f.value)}</p>

      {f.state === 'evidence' && f.quote ? (
        <>
          <blockquote {...stylex.props(text.bodySm, s.quote)}>“{f.quote}”</blockquote>
          {/* Not a <Label>: that uppercases, and a URL's path is case-sensitive. */}
          <p {...stylex.props(text.labelXs, s.note)}>
            FOUND IN{' '}
            {linkable(f.found_in) ? (
              <a href={f.found_in} target="_blank" rel="noopener noreferrer" {...stylex.props(s.link)}>
                {f.found_in}
              </a>
            ) : (
              <span>{f.found_in ?? 'an unrecorded location'}</span>
            )}
          </p>
        </>
      ) : null}
      {f.state === 'proposed' ? <p {...stylex.props(text.labelXs, s.note)}>PROPOSED BY THE MODEL: {f.basis ?? 'its reading of the source'}. Not a quote.</p> : null}
      {f.state === 'operator' && f.overrode?.provenance ? (
        <p {...stylex.props(text.labelXs, s.note)}>
          REPLACED THE MODEL&apos;S {String(f.overrode.provenance).toUpperCase()} VALUE: {displayValue(f.overrode.value as FieldState['value'])}
        </p>
      ) : null}
      {needs ? (
        <>
          <p {...stylex.props(text.bodySm, s.note)}>{f.why}</p>
          {f.searched ? <p {...stylex.props(text.bodySm, s.warn)}>ALREADY SEARCHED: {f.searched}</p> : null}
        </>
      ) : null}
      {f.rejected?.values?.length ? (
        <p {...stylex.props(text.labelXs, s.note)}>
          DROPPED: {f.rejected.values.join(', ')} ({f.rejected.why ?? 'rejected'})
        </p>
      ) : null}

      {editing ? (
        <form
          aria-label={`answer ${f.label}`}
          {...stylex.props(layout.stackSm)}
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          {multi ? (
            <div {...stylex.props(s.choices)}>
              {(f.choices ?? []).map((c) => (
                <label key={c} {...stylex.props(text.bodySm, s.choice)}>
                  <input
                    type="checkbox"
                    checked={picked.includes(c)}
                    onChange={(e) => setDraft(e.target.checked ? [...picked, c] : picked.filter((x) => x !== c))}
                  />
                  {c}
                </label>
              ))}
            </div>
          ) : (
            <input
              aria-label={f.label}
              value={typeof draft === 'string' ? draft : ''}
              onChange={(e) => setDraft(e.target.value)}
              {...stylex.props(text.bodySm, s.input)}
            />
          )}
          {f.field === 'licence' ? <Label>Type the licence as the source states it. &quot;unknown&quot; is not an answer.</Label> : null}
          <div {...stylex.props(layout.rowWrap)}>
            <button type="submit" {...stylex.props(text.labelMd, s.button)}>
              [ SAVE ]
            </button>
            <button type="button" onClick={() => setEditing(false)} {...stylex.props(text.labelMd, s.button, s.quiet)}>
              [ CANCEL ]
            </button>
          </div>
        </form>
      ) : null}
      <MsgLine msg={msg} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// The fields, the jobs and the actions for one dataset. Shared by the live
// card on NAEDOKO and the dataset detail screen.
// ---------------------------------------------------------------------------
export function ProvenanceBody({ d, reload }: { d: Dataset; reload: () => void }) {
  const { datasets } = useShared();
  const [msg, setMsg] = useState<Msg>(null);
  const clarify = d.stage === 'clarify';
  const fields = d.field_states ?? [];
  const needs = fields.filter((f) => f.state === 'needs_you').length;
  const assistLive = (d.assist_jobs ?? d.job_rows?.filter((j) => j.queue === 'dataset.assist') ?? []).some((j) => j.state === 'queued' || j.state === 'running');
  const jobsShown = (d.job_rows ?? []).filter((j) => j.state !== 'done' || j.queue === 'dataset.assist').slice(0, 6);

  const refresh = () => {
    reload();
    datasets.reload();
  };
  const advance = async () => {
    setMsg({ tone: 'busy', text: 'advancing' });
    const r = await postJson<{ ok: true; dataset: Dataset }>('/dash/api/dataset/advance', { id: d.id });
    setMsg(r.ok ? { tone: 'ok', text: `moved to ${r.data.dataset.stage}` } : { tone: 'err', text: `refused: ${refusalText(r)}` });
    refresh();
  };
  const askAgain = async () => {
    setMsg({ tone: 'busy', text: 'queueing the assist' });
    const r = await postJson<{ ok: true; job: string }>('/dash/api/dataset/assist', { id: d.id });
    setMsg(r.ok ? { tone: 'ok', text: `assist job ${r.data.job} queued on the gpu lane` } : { tone: 'err', text: `refused: ${refusalText(r)}` });
    refresh();
  };
  const onSaved = (r: Answered) => {
    refresh();
    // Answering the last open question is the operator's whole turn: the
    // pipeline takes it from there, exactly as it does after the assist.
    // A held dataset is the exception -- advancing it is a separate decision.
    if (!r.missing.length && r.dataset.stage === 'clarify' && !r.dataset.assist?.held) void advance();
  };

  if (!fields.length) {
    return (
      <StateView
        kind="error"
        title={`${PATHS.dataset(d.id)} sent no field_states`}
      />
    );
  }
  return (
    <div {...stylex.props(layout.stackSm)}>
      {jobsShown.length ? (
        <div {...stylex.props(layout.stack)} aria-label="jobs">
          {jobsShown.map((j) => (
            <Row key={j.id} bad={j.state === 'errored'}>
              <span {...stylex.props(layout.truncate)} title={jobLine(j)}>
                {jobLine(j)}
              </span>
              <Chip tone={j.state === 'errored' ? 'crimson' : j.state === 'done' ? 'moss' : j.state === 'running' ? 'cyan' : 'muted'}>
                {j.state} · {j.attempts}/{j.max_attempts}
              </Chip>
            </Row>
          ))}
        </div>
      ) : null}
      {(d.warnings ?? []).map((w, i) => (
        <p key={i} role="alert" {...stylex.props(text.bodySm, s.warn)}>
          {(w.kind ?? 'warning').toUpperCase()}: {w.what}
        </p>
      ))}
      {clarify && d.assist?.held ? <p {...stylex.props(text.bodySm, s.warn)}>{d.assist.held}</p> : null}
      <div {...stylex.props(s.fields)}>
        {fields.map((f) => (
          // Keyed on the value and state too, so a field the assist fills
          // while its editor is closed re-renders from the new truth.
          <FieldRow key={`${f.field}:${f.state}:${JSON.stringify(f.value)}`} datasetId={d.id} f={f} onSaved={onSaved} canEdit={clarify} />
        ))}
      </div>
      {!clarify ? <Label>Past clarify: provenance is fixed on the rows already written, so fields are read-only here.</Label> : null}
      {clarify ? (
        <div {...stylex.props(layout.rowWrap)}>
          {!needs && d.next_stage ? (
            <button type="button" onClick={() => void advance()} {...stylex.props(text.labelMd, s.button)}>
              [ ADVANCE TO {d.next_stage.toUpperCase()} ]
            </button>
          ) : null}
          {!assistLive ? (
            <button type="button" onClick={() => void askAgain()} {...stylex.props(text.labelMd, s.button, s.quiet)}>
              [ ASK THE MODEL AGAIN ]
            </button>
          ) : null}
        </div>
      ) : null}
      <MsgLine msg={msg} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// The live card: one dataset, polled fast while a job is moving.
// ---------------------------------------------------------------------------
export function DatasetCard({ id, onClose }: { id: string; onClose?: () => void }) {
  const { datasets } = useShared();
  const [every, setEvery] = useState(3000);
  const d = usePoll<Dataset>(PATHS.dataset(id), every);
  const want = pollEvery(d.data);
  useEffect(() => {
    setEvery(want);
  }, [want]);

  const close = onClose ? (
    <button type="button" onClick={onClose} aria-label={`stop watching ${id}`} {...stylex.props(text.labelXs, s.button, s.quiet)}>
      [ CLOSE ]
    </button>
  ) : undefined;
  if (!d.data) {
    return (
      <Panel kanji="芽" title={`DATASET ${id}`} tag="READING" tagTone="muted" flag={close}>
        <PollState path={PATHS.dataset(id)} failure={d.failure} what="dataset field" />
      </Panel>
    );
  }
  const x = d.data;
  const needs = (x.field_states ?? []).some((f) => f.state === 'needs_you');
  const errored = (x.job_rows ?? []).some((j) => j.state === 'errored');
  return (
    <Panel
      kanji="芽"
      title={x.name}
      tag={x.stage.toUpperCase()}
      tagTone={x.stage === 'complete' ? 'moss' : needs || errored ? 'crimson' : 'cyan'}
      edge={x.stage === 'complete' ? 'moss' : needs || errored ? 'crimson' : 'cyan'}
      sub={`${waitingOn(x)} · ${liveJobs(x).length ? 'polling every 3 s' : 'polling every 15 s'}`}
      flag={close}
      stale={d.stale}
    >
      {datasets.data ? <StageTrack o={datasets.data} current={x.stage} /> : null}
      <ProvenanceBody d={x} reload={d.reload} />
      <NavLink to={href.dataset(x.id)} {...stylex.props(text.labelXs, s.link)}>
        OPEN {x.id} →
      </NavLink>
    </Panel>
  );
}
