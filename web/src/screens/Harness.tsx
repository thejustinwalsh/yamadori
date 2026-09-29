// 道具箱 DOGUBAKO · HARNESS TOOLS -- what to put into a harness (prompts,
// skill folders, downloads, MCP servers, config, notes), each with its
// evidence, and the per-harness kit for the operator's own machine.
// GET/POST /dash/api/harness-kit (mcp/dash_harness.py).
import * as stylex from '@stylexjs/stylex';
import { useState, type ReactNode } from 'react';
import {
  ALL,
  byKind,
  downloadKit,
  emptyForm,
  fieldsOf,
  formOf,
  formProblems,
  HARNESS_DELETED_PATH,
  HARNESS_KIT_PATH,
  kitFileName,
  linkOf,
  POST,
  visible,
  when,
  type Entry,
  type Filter,
  type Form,
  type KitOverview,
  type Labelled,
  type Status,
  type Where,
} from '../api/harness';
import { invalidate } from '../api/cache';
import { usePoll } from '../api/usePoll';
import { colors, space } from '../tokens/tokens.stylex';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, type Tone } from '../ui/primitives';
import { PollState, StateView } from '../ui/StateView';
import { text } from '../ui/text';
import { Btn, Note, s as p, send, type Msg } from './skills/parts';

const h = stylex.create({
  bar: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 900px)': 'repeat(4, minmax(0, 1fr))' } },
  field: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 },
  check: { display: 'flex', alignItems: 'center', gap: space.spaceXs, color: colors.onSurfaceVariant, cursor: 'pointer', minHeight: '32px' },
  base: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere', textTransform: 'none' },
  card: {
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceXs,
    padding: space.spaceSm,
    minWidth: 0,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
  },
  cardOperator: { boxShadow: `inset 2px 0 0 ${colors.secondary}` },
  cardDeleted: { opacity: 0.6 },
  head: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: space.spaceXs, minWidth: 0 },
  title: { margin: 0, color: colors.primary, overflowWrap: 'anywhere', textTransform: 'none' },
  meta: { margin: 0, color: colors.outline, overflowWrap: 'anywhere', textTransform: 'none' },
  why: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere' },
  pre: { margin: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', color: colors.onSurface, maxHeight: '28rem', overflowY: 'auto', padding: space.spaceXs, backgroundColor: colors.surfaceContainerLow },
  list: { margin: 0, paddingInlineStart: '1.1rem', display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 },
  li: { overflowWrap: 'anywhere', color: colors.onSurface },
  code: { color: colors.tertiaryContainer, textTransform: 'none', overflowWrap: 'anywhere' },
  dl: { display: 'grid', gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 700px)': '9rem minmax(0, 1fr)' }, gap: '2px', columnGap: space.spaceSm, margin: 0 },
  dd: { margin: 0, overflowWrap: 'anywhere', color: colors.onSurface, textTransform: 'none' },
  grid2: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 700px)': 'repeat(2, minmax(0, 1fr))' } },
  grid3: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 700px)': 'repeat(3, minmax(0, 1fr))' } },
  fieldset: { margin: 0, padding: space.spaceSm, borderWidth: 1, borderStyle: 'dashed', borderColor: colors.outlineVariant, minWidth: 0, display: 'flex', flexDirection: 'column', gap: space.spaceSm },
  legend: { color: colors.outline, paddingInline: space.spaceXs },
});

const STATUS_TONE: Record<Status, Tone> = { recommended: 'moss', trying: 'cyan', rejected: 'crimson' };
const PROOF_LABEL: Record<string, string> = { operator: 'OPERATOR', measured: 'MEASURED', verified: 'VERIFIED · NO MODEL', unmeasured: 'UNMEASURED' };
const WHERE_LABEL: Record<Where, string> = { remote: 'REMOTE MACHINE', box: 'LOCAL BOX', both: 'REMOTE + BOX' };

function labelOf(list: Labelled[], id: string): string {
  return list.find((x) => x.id === id)?.label ?? id;
}

export function EntryCard({
  e,
  harnesses,
  onEdit,
  onDelete,
  onRestore,
  busy,
}: {
  e: Entry;
  harnesses: Labelled[];
  onEdit?: () => void;
  onDelete?: () => void;
  onRestore?: () => void;
  busy?: boolean;
}) {
  const d = e.download;
  return (
    <article data-entry={e.id} aria-label={e.title} {...stylex.props(h.card, e.needs_operator && h.cardOperator, e.deleted && h.cardDeleted)}>
      <div {...stylex.props(h.head)}>
        {e.needs_operator ? <Chip tone="rose">NEEDS OPERATOR</Chip> : null}
        <Chip tone={STATUS_TONE[e.status] ?? 'muted'}>{e.status.toUpperCase()}</Chip>
        <Chip tone={e.proof === 'unmeasured' ? 'rose' : 'muted'} title="what the claim rests on">
          {PROOF_LABEL[e.proof] ?? e.proof}
        </Chip>
        <Chip tone="muted">{WHERE_LABEL[e.where] ?? e.where}</Chip>
        <Chip tone={e.harness === 'any' ? 'cyan' : 'muted'}>{labelOf(harnesses, e.harness).toUpperCase()}</Chip>
        {e.deleted ? <Chip tone="crimson">DELETED</Chip> : null}
      </div>
      <h3 {...stylex.props(text.titleMd, h.title)}>{e.title}</h3>
      {e.status_why ? (
        <p {...stylex.props(text.bodySm, h.why)}>
          <Label>WHY {e.status.toUpperCase()}</Label> {e.status_why}
        </p>
      ) : null}
      {e.body ? (
        e.body.length > 600 || e.kind === 'config' || e.kind === 'mcp_server' || e.kind === 'prompt' ? (
          <details>
            <summary {...stylex.props(text.labelXs, p.summary)}>
              {e.kind === 'note' ? 'TEXT' : 'CONTENT'} · {e.body.length.toLocaleString()} chars{e.file_name ? ` · ${e.file_name}` : ''}
            </summary>
            <pre {...stylex.props(text.bodySm, h.pre)}>{e.body}</pre>
          </details>
        ) : (
          <pre {...stylex.props(text.bodySm, h.pre)}>{e.body}</pre>
        )
      ) : null}
      {e.skill_path || e.target ? (
        <dl {...stylex.props(h.dl)}>
          {e.skill_path ? (
            <>
              <dt><Label>SKILL FOLDER</Label></dt>
              <dd {...stylex.props(text.bodySm, h.dd, h.code)}>{e.skill_path}</dd>
            </>
          ) : null}
          {e.target ? (
            <>
              <dt><Label>GOES TO</Label></dt>
              <dd {...stylex.props(text.bodySm, h.dd)}>{e.target}</dd>
            </>
          ) : null}
        </dl>
      ) : null}
      {d ? (
        <dl aria-label="download" {...stylex.props(h.dl)}>
          {d.source_url ? (
            <>
              <dt><Label>SOURCE</Label></dt>
              <dd {...stylex.props(text.bodySm, h.dd)}>
                <a href={d.source_url} target="_blank" rel="noreferrer noopener" {...stylex.props(p.link)}>{d.source_url}</a>
              </dd>
            </>
          ) : null}
          <dt><Label>VERSION</Label></dt>
          <dd {...stylex.props(text.bodySm, h.dd, h.code)}>{d.version || '—'}</dd>
          <dt><Label>PIN</Label></dt>
          <dd {...stylex.props(text.bodySm, h.dd, h.code)}>{d.hash || (d.commit ? `commit ${d.commit}` : `none: ${d.pin_missing_why || 'not given'}`)}</dd>
          <dt><Label>LICENCE</Label></dt>
          <dd {...stylex.props(text.bodySm, h.dd)}>{d.licence || '—'}</dd>
          {(['windows', 'unix', 'any'] as const).map((k) =>
            d.install?.[k] ? (
              <InstallRow key={k} os={k} cmd={d.install[k] as string} />
            ) : null,
          )}
          {d.verify ? (
            <>
              <dt><Label>VERIFY</Label></dt>
              <dd {...stylex.props(text.bodySm, h.dd, h.code)}>{d.verify}</dd>
            </>
          ) : null}
        </dl>
      ) : null}
      {e.evidence.length ? (
        <div>
          <Label>EVIDENCE</Label>
          <ul {...stylex.props(h.list)}>
            {e.evidence.map((x, i) => {
              const url = linkOf(x.ref);
              return (
                <li key={i} {...stylex.props(text.bodySm, h.li)}>
                  {url ? (
                    <a href={url} target="_blank" rel="noreferrer noopener" {...stylex.props(p.link)}>{x.ref}</a>
                  ) : (
                    <span {...stylex.props(h.code)}>{x.ref}</span>
                  )}{' '}
                  — {x.showed}
                </li>
              );
            })}
          </ul>
        </div>
      ) : (
        <p {...stylex.props(text.labelXs, h.meta)}>NO EVIDENCE YET</p>
      )}
      <p {...stylex.props(text.labelXs, h.meta)}>
        v{e.version} · updated {when(e.updated)} by {e.updated_by}
        {e.deleted ? ` · deleted ${when(e.deleted_at)} by ${e.deleted_by}: ${e.deleted_why}` : ''}
      </p>
      {onEdit || onDelete || onRestore ? (
        <div {...stylex.props(layout.rowWrap)}>
          {onEdit ? <Btn onClick={onEdit} disabled={busy}>EDIT</Btn> : null}
          {onDelete ? <Btn onClick={onDelete} disabled={busy} danger>DELETE</Btn> : null}
          {onRestore ? <Btn onClick={onRestore} disabled={busy}>RESTORE</Btn> : null}
        </div>
      ) : null}
    </article>
  );
}

function InstallRow({ os, cmd }: { os: 'windows' | 'unix' | 'any'; cmd: string }) {
  return (
    <>
      <dt><Label>INSTALL · {os === 'unix' ? 'LINUX/MAC' : os.toUpperCase()}</Label></dt>
      <dd {...stylex.props(text.bodySm, h.dd, h.code)}>{cmd}</dd>
    </>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label {...stylex.props(h.field)}>
      <Label>{label}</Label>
      {children}
    </label>
  );
}

/** The add / edit form. Controlled: `onChange` gets the next form. */
export function EntryForm({
  form,
  vocab,
  onChange,
  onSubmit,
  onCancel,
  busy,
  msg,
  editing,
}: {
  form: Form;
  vocab: Pick<KitOverview, 'harnesses' | 'kinds' | 'statuses' | 'proofs' | 'wheres' | 'skill_roots' | 'tokens' | 'key_placeholder'>;
  onChange: (f: Form) => void;
  onSubmit: () => void;
  onCancel: () => void;
  busy?: boolean;
  msg?: Msg;
  editing?: string | null;
}) {
  const set = <K extends keyof Form>(k: K, v: Form[K]) => onChange({ ...form, [k]: v });
  const setDl = (k: keyof Form['dl'], v: string) => onChange({ ...form, dl: { ...form.dl, [k]: v } });
  const problems = formProblems(form);
  const showDl = form.kind === 'download' || form.kind === 'mcp_server' || Object.values(form.dl).some((v) => v);
  return (
    <form
      aria-label={editing ? 'edit entry' : 'add entry'}
      onSubmit={(ev) => {
        ev.preventDefault();
        onSubmit();
      }}
      {...stylex.props(layout.stackSm)}
    >
      <div {...stylex.props(h.grid3)}>
        <Field label="HARNESS">
          <select value={form.harness} onChange={(e) => set('harness', e.target.value)} {...stylex.props(text.bodySm, p.input)}>
            {vocab.harnesses.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </Field>
        <Field label="KIND">
          <select value={form.kind} onChange={(e) => set('kind', e.target.value)} {...stylex.props(text.bodySm, p.input)}>
            {vocab.kinds.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </Field>
        <Field label="APPLIES ON">
          <select value={form.where} onChange={(e) => set('where', e.target.value as Where)} {...stylex.props(text.bodySm, p.input)}>
            {vocab.wheres.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </Field>
      </div>
      <Field label="TITLE">
        <input value={form.title} onChange={(e) => set('title', e.target.value)} {...stylex.props(text.bodySm, p.input)} />
      </Field>
      <Field label={`BODY (tokens the kit fills: ${vocab.tokens.join(' ')})`}>
        <textarea value={form.body} onChange={(e) => set('body', e.target.value)} {...stylex.props(text.bodySm, p.input, p.area)} />
      </Field>
      <div {...stylex.props(h.grid3)}>
        <Field label="STATUS">
          <select value={form.status} onChange={(e) => set('status', e.target.value as Status)} {...stylex.props(text.bodySm, p.input)}>
            {vocab.statuses.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </Field>
        <Field label="PROOF">
          <select value={form.proof} onChange={(e) => set('proof', e.target.value as Form['proof'])} {...stylex.props(text.bodySm, p.input)}>
            {vocab.proofs.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </Field>
        <label {...stylex.props(text.labelXs, h.check)}>
          <input type="checkbox" checked={form.needs_operator} onChange={(e) => set('needs_operator', e.target.checked)} /> NEEDS OPERATOR
        </label>
      </div>
      <Field label={`WHY ${form.status.toUpperCase()}${form.status === 'recommended' ? ' (optional)' : ''}`}>
        <input value={form.status_why} onChange={(e) => set('status_why', e.target.value)} {...stylex.props(text.bodySm, p.input)} />
      </Field>
      <div {...stylex.props(h.grid3)}>
        <Field label="GOES TO (where on the machine)">
          <input value={form.target} onChange={(e) => set('target', e.target.value)} {...stylex.props(text.bodySm, p.input)} />
        </Field>
        <Field label="FILE NAME IN THE KIT">
          <input value={form.file_name} onChange={(e) => set('file_name', e.target.value)} placeholder="models.json" {...stylex.props(text.bodySm, p.input)} />
        </Field>
        {form.kind === 'skill' ? (
          <Field label={`SKILL FOLDER (under ${vocab.skill_roots.join(' or ')})`}>
            <input value={form.skill_path} onChange={(e) => set('skill_path', e.target.value)} {...stylex.props(text.bodySm, p.input)} />
          </Field>
        ) : null}
      </div>
      <fieldset {...stylex.props(h.fieldset)}>
        <legend {...stylex.props(text.labelXs, h.legend)}>EVIDENCE · a doc path, run name or link, and what it showed</legend>
        {form.evidence.map((x, i) => (
          <div key={i} {...stylex.props(h.grid2)}>
            <input aria-label={`evidence ${i + 1} ref`} value={x.ref} placeholder="docs/HARNESSES.md s0" onChange={(e) => set('evidence', form.evidence.map((y, j) => (j === i ? { ...y, ref: e.target.value } : y)))} {...stylex.props(text.bodySm, p.input)} />
            <input aria-label={`evidence ${i + 1} showed`} value={x.showed} placeholder="what it showed (with its n)" onChange={(e) => set('evidence', form.evidence.map((y, j) => (j === i ? { ...y, showed: e.target.value } : y)))} {...stylex.props(text.bodySm, p.input)} />
          </div>
        ))}
        <div {...stylex.props(layout.rowWrap)}>
          <Btn onClick={() => set('evidence', [...form.evidence, { ref: '', showed: '' }])}>ADD EVIDENCE</Btn>
          {form.evidence.length > 1 ? <Btn onClick={() => set('evidence', form.evidence.slice(0, -1))}>REMOVE LAST</Btn> : null}
        </div>
      </fieldset>
      {showDl ? (
        <fieldset {...stylex.props(h.fieldset)}>
          <legend {...stylex.props(text.labelXs, h.legend)}>DOWNLOAD · exact version, sha256 / sha512 or commit, licence, install per OS</legend>
          <div {...stylex.props(h.grid3)}>
            <Field label="SOURCE URL"><input value={form.dl.source_url} onChange={(e) => setDl('source_url', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
            <Field label="EXACT VERSION"><input value={form.dl.version} onChange={(e) => setDl('version', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
            <Field label="LICENCE"><input value={form.dl.licence} onChange={(e) => setDl('licence', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
          </div>
          <div {...stylex.props(h.grid2)}>
            <Field label="HASH (sha256:<hex> or sha512-<base64>)"><input value={form.dl.hash} onChange={(e) => setDl('hash', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
            <Field label="COMMIT (40 hex)"><input value={form.dl.commit} onChange={(e) => setDl('commit', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
          </div>
          <Field label="NO PIN YET BECAUSE (when there is no hash or commit)"><input value={form.dl.pin_missing_why} onChange={(e) => setDl('pin_missing_why', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
          <div {...stylex.props(h.grid3)}>
            <Field label="INSTALL · WINDOWS"><input value={form.dl.windows} onChange={(e) => setDl('windows', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
            <Field label="INSTALL · LINUX/MAC"><input value={form.dl.unix} onChange={(e) => setDl('unix', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
            <Field label="INSTALL · ANY OS"><input value={form.dl.any} onChange={(e) => setDl('any', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
          </div>
          <Field label="VERIFY (a command that checks the pin)"><input value={form.dl.verify} onChange={(e) => setDl('verify', e.target.value)} {...stylex.props(text.bodySm, p.input)} /></Field>
        </fieldset>
      ) : null}
      <p {...stylex.props(text.bodySm, p.why)}>
        Never put a key here: write an environment variable or {vocab.key_placeholder}. A key-shaped value is refused.
      </p>
      {problems.length ? (
        <p role="status" {...stylex.props(text.bodySm, p.why)}>Still needed: {problems.join('; ')}.</p>
      ) : null}
      <div {...stylex.props(layout.rowWrap)}>
        <Btn type="submit" disabled={busy || problems.length > 0}>{editing ? 'SAVE NEW VERSION' : 'ADD ENTRY'}</Btn>
        <Btn onClick={onCancel} disabled={busy}>CANCEL</Btn>
        <Note msg={msg ?? null} />
      </div>
    </form>
  );
}

function Toolbar({
  data,
  filter,
  setFilter,
  showDeleted,
  setShowDeleted,
  onAdd,
}: {
  data: KitOverview;
  filter: Filter;
  setFilter: (f: Filter) => void;
  showDeleted: boolean;
  setShowDeleted: (v: boolean) => void;
  onAdd: () => void;
}) {
  const [kitMsg, setKitMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const exportable = filter.harness !== ALL;
  const shown = visible(data.entries, filter).length;
  return (
    <Panel kanji="道具箱" title="HARNESS TOOLS" tag={`${data.counts.total} ENTRIES`} sub="what to put into each harness: prompts, skills, downloads, MCP servers, config, notes -- each with its evidence" edge="moss">
      <div {...stylex.props(h.bar)}>
        <Field label="HARNESS">
          <select aria-label="harness" value={filter.harness} onChange={(e) => setFilter({ ...filter, harness: e.target.value })} {...stylex.props(text.bodySm, p.input)}>
            <option value={ALL}>all harnesses</option>
            {data.harnesses.map((x) => (
              <option key={x.id} value={x.id}>
                {x.id === 'any' ? 'general (any harness)' : x.label} · {data.counts.by_harness[x.id] ?? 0}
              </option>
            ))}
          </select>
        </Field>
        <Field label="STATUS">
          <select aria-label="status" value={filter.status} onChange={(e) => setFilter({ ...filter, status: e.target.value as Status | '' })} {...stylex.props(text.bodySm, p.input)}>
            <option value="">every status</option>
            {data.statuses.map((x) => <option key={x} value={x}>{x} · {data.counts.by_status[x] ?? 0}</option>)}
          </select>
        </Field>
        <Field label="APPLIES ON">
          <select aria-label="applies on" value={filter.where} onChange={(e) => setFilter({ ...filter, where: e.target.value as Where | '' })} {...stylex.props(text.bodySm, p.input)}>
            <option value="">either machine</option>
            {data.wheres.filter((x) => x.id !== 'both').map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </Field>
        <Field label="SEARCH">
          <input type="search" aria-label="search entries" value={filter.q} onChange={(e) => setFilter({ ...filter, q: e.target.value })} placeholder="title, text, evidence" {...stylex.props(text.bodySm, p.input)} />
        </Field>
      </div>
      <div {...stylex.props(layout.rowWrap)}>
        <label {...stylex.props(text.labelXs, h.check)}>
          <input type="checkbox" checked={filter.needsOperator} onChange={(e) => setFilter({ ...filter, needsOperator: e.target.checked })} /> NEEDS OPERATOR ONLY · {data.counts.needs_operator}
        </label>
        <label {...stylex.props(text.labelXs, h.check)}>
          <input type="checkbox" checked={showDeleted} onChange={(e) => setShowDeleted(e.target.checked)} /> SHOW DELETED
        </label>
        <Btn onClick={onAdd}>ADD ENTRY</Btn>
        <Btn
          disabled={!exportable || busy}
          title={exportable ? `the ${filter.harness} kit for the remote machine` : 'pick a harness to export its kit'}
          onClick={() => {
            setBusy(true);
            setKitMsg({ tone: 'ok', text: 'building the kit' });
            void downloadKit(filter.harness).then((err) => {
              setBusy(false);
              setKitMsg(err ? { tone: 'err', text: `kit not downloaded: ${err}` } : { tone: 'ok', text: `downloaded ${kitFileName(filter.harness)}` });
            });
          }}
        >
          DOWNLOAD KIT{exportable ? ` · ${filter.harness.toUpperCase()}` : ''}
        </Btn>
        <Note msg={kitMsg} />
      </div>
      <p {...stylex.props(text.bodySm, h.base)}>
        The kit is for the machine the next waves run on: its config points at <span {...stylex.props(h.code)}>{data.public_base.url}</span> ({data.public_base.source}), it carries the recommended skills, prompts and pinned install
        commands, and <b>no key</b> -- {data.key_placeholder} and the README say where yours goes. Also at GET /dash/api/harness-kit/export/&lt;harness&gt;. Showing {shown} of {data.counts.total}.
      </p>
    </Panel>
  );
}

function DeletedPanel({ harnesses, onRestore, busy }: { harnesses: Labelled[]; onRestore: (e: Entry) => void; busy: boolean }) {
  const d = usePoll<{ entries: Entry[] }>(HARNESS_DELETED_PATH, 0);
  return (
    <Panel kanji="削" title="DELETED" tag="SOFT: KEPT WITH HISTORY" tagTone="muted">
      {d.data ? (
        d.data.entries.length ? (
          <div {...stylex.props(layout.stackSm)}>
            {d.data.entries.map((e) => (
              <EntryCard key={e.id} e={e} harnesses={harnesses} busy={busy} onRestore={() => onRestore(e)} />
            ))}
          </div>
        ) : (
          <StateView kind="empty" title="nothing deleted" />
        )
      ) : (
        <PollState path={HARNESS_DELETED_PATH} failure={d.failure} />
      )}
    </Panel>
  );
}

export function Harness() {
  const poll = usePoll<KitOverview>(HARNESS_KIT_PATH, 30000);
  const [filter, setFilter] = useState<Filter>({ harness: ALL, status: '', where: '', needsOperator: false, q: '' });
  const [showDeleted, setShowDeleted] = useState(false);
  const [editing, setEditing] = useState<{ id: string | null; version: number | null; form: Form } | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const [cardMsg, setCardMsg] = useState<Msg>(null);
  const [deletedNonce, setDeletedNonce] = useState(0);
  const data = poll.data;
  if (!data) return <PollState path={HARNESS_KIT_PATH} failure={poll.failure} />;

  const refresh = () => {
    invalidate(HARNESS_DELETED_PATH);
    setDeletedNonce((n) => n + 1);
    poll.reload();
  };
  const save = async () => {
    if (!editing) return;
    setBusy(true);
    const fields = fieldsOf(editing.form);
    const r = editing.id
      ? await send(POST.edit, { id: editing.id, base_version: editing.version, fields }, 'saved as a new version')
      : await send(POST.create, { fields }, 'added');
    setBusy(false);
    setMsg(r.msg);
    if (r.data) {
      setEditing(null);
      setCardMsg(r.msg);
      refresh();
    }
  };
  const del = async (e: Entry) => {
    const why = window.prompt(`Delete "${e.title}"? It is kept, with its history. Why?`);
    if (!why) return;
    setBusy(true);
    const r = await send(POST.del, { id: e.id, why }, 'deleted (kept with its history)');
    setBusy(false);
    setCardMsg(r.msg);
    refresh();
  };
  const restore = async (e: Entry) => {
    setBusy(true);
    const r = await send(POST.restore, { id: e.id }, 'restored');
    setBusy(false);
    setCardMsg(r.msg);
    refresh();
  };

  const groups = byKind(visible(data.entries, filter), data.kinds);
  return (
    <>
      <Toolbar
        data={data}
        filter={filter}
        setFilter={setFilter}
        showDeleted={showDeleted}
        setShowDeleted={setShowDeleted}
        onAdd={() => {
          setMsg(null);
          setEditing({ id: null, version: null, form: emptyForm(filter.harness) });
        }}
      />
      {editing ? (
        <Panel kanji="記" title={editing.id ? 'EDIT ENTRY' : 'ADD ENTRY'} tag={editing.id ? `FROM v${editing.version}` : 'NEW'} tagTone="cyan" edge="cyan">
          <EntryForm
            form={editing.form}
            vocab={data}
            editing={editing.id}
            busy={busy}
            msg={msg}
            onChange={(form) => setEditing({ ...editing, form })}
            onSubmit={() => void save()}
            onCancel={() => setEditing(null)}
          />
        </Panel>
      ) : null}
      {cardMsg ? <Note msg={cardMsg} /> : null}
      {groups.length === 0 ? <StateView kind="empty" title="no entry matches this filter" /> : null}
      {groups.map((g) => (
        <ErrorBoundary key={g.kind.id} what={g.kind.label.toUpperCase()} source={HARNESS_KIT_PATH}>
          <Panel title={g.kind.label.toUpperCase()} tag={`${g.entries.length}`} tagTone="muted" id={`kind-${g.kind.id}`}>
            <div {...stylex.props(layout.stackSm)}>
              {g.entries.map((e) => (
                <EntryCard
                  key={e.id}
                  e={e}
                  harnesses={data.harnesses}
                  busy={busy}
                  onEdit={() => {
                    setMsg(null);
                    setEditing({ id: e.id, version: e.version, form: formOf(e) });
                    window.scrollTo(0, 0);
                  }}
                  onDelete={() => void del(e)}
                />
              ))}
            </div>
          </Panel>
        </ErrorBoundary>
      ))}
      {showDeleted ? <DeletedPanel key={deletedNonce} harnesses={data.harnesses} busy={busy} onRestore={(e) => void restore(e)} /> : null}
    </>
  );
}
