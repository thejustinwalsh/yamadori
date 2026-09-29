// Parts the skill factory's screens share: controls, the sub-navigation,
// the stage rail, the size bar and a small SKILL.md renderer. Every number
// drawn here comes from the payload it is handed.
import * as stylex from '@stylexjs/stylex';
import { Fragment, type ReactNode } from 'react';
import { postJson } from '../../api/client';
import { overCaps, sizeTone, stageStates, type Limits, type Size, type SkillJob, type SkillVersion, type StageState } from '../../api/skills';
import { href, SKILLS_VIEWS, type SkillsView } from '../../router';
import { colors, space } from '../../tokens/tokens.stylex';
import { NavLink } from '../../ui/NavLink';
import { Chip, Label, layout, Meter, type Tone } from '../../ui/primitives';
import { text } from '../../ui/text';

export const s = stylex.create({
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
  areaTall: { minHeight: '22rem' },
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
  buttonDanger: {
    color: colors.secondary,
    borderColor: colors.secondary,
    backgroundColor: { default: colors.surfaceContainerLowest, ':hover': `color-mix(in srgb, ${colors.secondaryContainer} 18%, transparent)` },
  },
  link: { color: colors.tertiaryContainer, textDecoration: 'none', overflowWrap: 'anywhere', ':hover': { textDecoration: 'underline' } },
  pre: { margin: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', color: colors.onSurface, maxHeight: '32rem', overflowY: 'auto' },
  why: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere', textTransform: 'none' },
  err: { margin: 0, color: colors.secondary, overflowWrap: 'anywhere', textTransform: 'none' },
  ok: { margin: 0, color: colors.primaryContainer, overflowWrap: 'anywhere' },
  quote: {
    margin: 0,
    paddingInlineStart: space.spaceSm,
    borderInlineStartWidth: 2,
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: colors.outlineVariant,
    color: colors.onSurfaceVariant,
    overflowWrap: 'anywhere',
    fontStyle: 'italic',
  },
  summary: { cursor: 'pointer', color: colors.outline },
  field: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0, flexGrow: 1 },
  tabs: { display: 'flex', flexWrap: 'wrap', gap: '2px', minWidth: 0 },
  tab: {
    textDecoration: 'none',
    color: colors.outline,
    paddingInline: space.spaceSm,
    paddingBlock: space.spaceXs,
    borderBottomWidth: 2,
    borderBottomStyle: 'solid',
    borderBottomColor: 'transparent',
    backgroundColor: colors.surfaceContainerLowest,
    ':hover': { color: colors.primary },
  },
  tabOn: { color: colors.primaryContainer, borderBottomColor: colors.primaryContainer },
  rail: { display: 'flex', gap: '2px', overflowX: 'auto', minWidth: 0 },
  cell: {
    flexGrow: 1,
    flexBasis: 0,
    minWidth: '64px',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    padding: space.spaceXs,
    backgroundColor: colors.surfaceContainerLowest,
    borderTopWidth: 2,
    borderTopStyle: 'solid',
    borderTopColor: colors.outlineVariant,
  },
  cellSkipped: { opacity: 0.35, borderTopStyle: 'dashed' },
  md: { display: 'flex', flexDirection: 'column', gap: space.spaceXs, color: colors.onSurface, overflowWrap: 'anywhere' },
  mdH: { margin: 0, color: colors.primary },
  mdP: { margin: 0 },
  mdList: { margin: 0, paddingInlineStart: '1.2rem', display: 'flex', flexDirection: 'column', gap: '2px' },
  code: { color: colors.tertiaryContainer, backgroundColor: colors.surfaceContainerLowest, paddingInline: '2px' },
  form: { color: colors.primaryContainer, fontWeight: 700 },
  formNot: { color: colors.secondary, fontWeight: 700 },
  // API names and topics are case-sensitive: never uppercased by a label style.
  asIs: { textTransform: 'none' },
});

export type Msg = { tone: 'ok' | 'err'; text: string } | null;

export function Note({ msg }: { msg: Msg }) {
  if (!msg) return null;
  return (
    <p role={msg.tone === 'err' ? 'alert' : 'status'} {...stylex.props(text.bodySm, msg.tone === 'err' ? s.err : s.ok)}>
      {msg.tone === 'err' ? '[ ! ] ' : ''}
      {msg.text}
    </p>
  );
}

/** POST and say what happened, in the server's own words when it refused. */
export async function send<T = unknown>(path: string, body: unknown, ok = 'saved'): Promise<{ msg: Msg; data: T | null }> {
  const r = await postJson<T>(path, body);
  if (r.ok) return { msg: { tone: 'ok', text: ok }, data: r.data };
  const f = r.failure;
  return { msg: { tone: 'err', text: 'message' in f ? f.message : f.kind }, data: null };
}

export function Btn({
  children,
  onClick,
  disabled,
  danger,
  type = 'button',
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  danger?: boolean;
  type?: 'button' | 'submit';
  title?: string;
}) {
  return (
    <button type={type} title={title} disabled={disabled} onClick={onClick} {...stylex.props(text.labelMd, s.button, danger && s.buttonDanger)}>
      [ {children} ]
    </button>
  );
}

export function Field({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <label {...stylex.props(s.field)}>
      <Label>{label}</Label>
      {children}
    </label>
  );
}

const VIEW_LABEL: Record<SkillsView, string> = { library: 'LIBRARY', create: 'CREATE', selections: 'SELECTIONS', prompts: 'PROMPTS' };

/** LIBRARY · CREATE · SELECTIONS · PROMPTS. `current` is null on a skill's own page. */
export function SubNav({ current, extra }: { current: SkillsView | null; extra?: ReactNode }) {
  return (
    <nav aria-label="skill factory" {...stylex.props(layout.between)}>
      <div {...stylex.props(s.tabs)}>
        {SKILLS_VIEWS.map((v) => (
          <NavLink key={v} to={href.skillsView(v)} aria-current={v === current ? 'page' : undefined} {...stylex.props(text.labelMd, s.tab, v === current && s.tabOn)}>
            {VIEW_LABEL[v]}
          </NavLink>
        ))}
      </div>
      {extra}
    </nav>
  );
}

export const STAGE_TONE: Record<StageState, Tone | null> = {
  done: 'moss',
  armed: 'moss',
  running: 'cyan',
  queued: 'cyan',
  failed: 'crimson',
  quarantined: 'crimson',
  decomposed: 'cyan',
  pending: null,
  skipped: null,
};

const railTone = stylex.create({
  moss: { borderTopColor: colors.primaryContainer },
  cyan: { borderTopColor: colors.tertiaryContainer, boxShadow: `inset 0 12px 16px -12px color-mix(in srgb, ${colors.tertiaryContainer} 35%, transparent)` },
  crimson: { borderTopColor: colors.secondaryContainer, boxShadow: `inset 0 12px 16px -12px color-mix(in srgb, ${colors.secondaryContainer} 45%, transparent)` },
  rose: {},
  muted: {},
});

/**
 * One version's walk over the pipeline's stages (the overview's `stages`):
 * each cell is done, where the version stopped (with its state), waiting,
 * or not on this version's path. Not a progress bar.
 */
export function StageRail({ stages, version, jobs, prompts }: { stages: string[]; version: Pick<SkillVersion, 'path' | 'stage' | 'state'>; jobs?: SkillJob[]; prompts?: Record<string, string> }) {
  const cells = stageStates(stages, version, jobs);
  return (
    <div role="list" aria-label="pipeline stages" {...stylex.props(s.rail)}>
      {cells.map((c, i) => {
        const tone = STAGE_TONE[c.state];
        return (
          <div key={c.stage} role="listitem" {...stylex.props(s.cell, tone && railTone[tone], c.state === 'skipped' && s.cellSkipped)}>
            <Label>{String(i + 1).padStart(2, '0')}</Label>
            <span {...stylex.props(text.labelXs, c.state === 'pending' || c.state === 'skipped' ? text.outline : text.onSurface)}>{c.stage}</span>
            <span {...stylex.props(text.labelXs, tone === 'crimson' ? text.rose : tone === 'cyan' ? text.cyan : tone === 'moss' ? text.moss : text.outline)}>
              {c.state === 'skipped' ? 'not on path' : c.state === 'pending' ? (version.state === 'planned' ? 'on path' : 'waits') : c.state}
            </span>
            {prompts?.[c.stage] && c.state !== 'skipped' ? <span {...stylex.props(text.labelXs, text.outline)}>{prompts[c.stage]}</span> : null}
          </div>
        );
      })}
    </div>
  );
}

/** The injected body against the per-skill caps, with every other cap the size breaks. */
export function SizeBar({ size, limits, compact }: { size: Size | null | undefined; limits: Limits; compact?: boolean }) {
  if (!size) return <span {...stylex.props(text.labelXs, text.outline)}>—</span>;
  const tone = sizeTone(size.tokens, limits);
  const over = overCaps(size, limits);
  const label = `${size.tokens} of ${limits.skill_tokens_hard} tokens (aim ${limits.skill_tokens_aim[0]}-${limits.skill_tokens_aim[1]})`;
  return (
    <div {...stylex.props(layout.stack)} title={over.length ? over.join('; ') : label}>
      <Meter value={size.tokens / limits.skill_tokens_hard} tone={tone} label={label} />
      <span {...stylex.props(text.labelXs, tone === 'crimson' ? text.rose : text.outline, text.num)}>
        {size.tokens}/{limits.skill_tokens_hard} TOK
        {compact ? '' : ` · ${size.items}/${limits.max_items} ITEMS · ${size.prohibitions}/${limits.max_prohibitions} DO NOT`}
      </span>
      {!compact && over.length ? <span {...stylex.props(text.bodySm, s.err)}>{over.join('; ')}</span> : null}
    </div>
  );
}

/** The prohibition count against the cap, as a chip. */
export function ProhibitionChip({ n, max }: { n: number | undefined; max: number }) {
  if (n === undefined) return <span {...stylex.props(text.labelXs, text.outline)}>—</span>;
  return (
    <Chip tone={n > max ? 'crimson' : n === max ? 'rose' : n ? 'muted' : 'moss'} title={`${n} prohibition(s); the cap is ${max}`}>
      {n}/{max}
    </Chip>
  );
}

export function Chips({ values, names, tone = 'muted', empty = '—' }: { values: string[] | undefined; names?: Record<string, string>; tone?: Tone; empty?: string }) {
  if (!values?.length) return <span {...stylex.props(text.labelXs, text.outline)}>{empty}</span>;
  return (
    <span {...stylex.props(layout.rowWrap)}>
      {values.map((v) => (
        <Chip key={v} tone={tone} title={names?.[v] && names[v] !== v ? v : undefined}>
          <span {...stylex.props(s.asIs)}>{names?.[v] ?? v}</span>
        </Chip>
      ))}
    </span>
  );
}

/** A JSON value under a disclosure: every stage output, verbatim. */
export function Raw({ label, value }: { label: ReactNode; value: unknown }) {
  if (value === null || value === undefined) return null;
  return (
    <details>
      <summary {...stylex.props(text.labelXs, s.summary)}>{label}</summary>
      <pre {...stylex.props(text.bodySm, s.pre)}>{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

// ---------------------------------------------------------------------------
// A small markdown renderer for SKILL.md bodies: headings, lists, fences,
// paragraphs, inline code and bold. React elements only -- the text is data
// and is never parsed as HTML.
// ---------------------------------------------------------------------------
function inline(t: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(t))) {
    if (m.index > last) out.push(t.slice(last, m.index));
    const tok = m[0];
    out.push(
      tok.startsWith('`') ? (
        <code key={`${key}c${i++}`} {...stylex.props(s.code)}>
          {tok.slice(1, -1)}
        </code>
      ) : (
        <strong key={`${key}b${i++}`}>{tok.slice(2, -2)}</strong>
      ),
    );
    last = m.index + tok.length;
  }
  if (last < t.length) out.push(t.slice(last));
  return out;
}

/** A guidance line: its DO / WHEN x / DO NOT form set apart. */
function itemLine(t: string, key: string): ReactNode {
  const m = /^(DO NOT|DON'T|NEVER|DO|WHEN [^:]+)\s*:\s*(.*)$/s.exec(t);
  if (!m) return inline(t, key);
  const form = m[1] ?? '';
  const neg = /^(DO NOT|DON'T|NEVER)/.test(form);
  return (
    <>
      <span {...stylex.props(neg ? s.formNot : s.form)}>{form}:</span> {inline(m[2] ?? '', key)}
    </>
  );
}

const LIST = /^\s*([-*]|\d+\.)\s+/;

/** The blocks of a markdown text. A plain function, not a component: its
 * loop counters are not React state. */
function mdBlocks(md: string): ReactNode[] {
  const blocks: ReactNode[] = [];
  const lines = md.replace(/\r\n/g, '\n').split('\n');
  const at = (j: number) => lines[j] ?? '';
  let i = 0;
  let k = 0;
  while (i < lines.length) {
    const line = at(i);
    if (/^```/.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(at(i))) body.push(at(i++));
      i++;
      blocks.push(
        <pre key={k++} {...stylex.props(text.bodySm, s.pre, s.code)}>
          {body.join('\n')}
        </pre>,
      );
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const level = (h[1] ?? '#').length;
      blocks.push(
        <p key={k++} role="heading" aria-level={Math.min(6, level + 2)} {...stylex.props(level === 1 ? text.titleMd : text.labelMd, s.mdH)}>
          {inline(h[2] ?? '', `h${k}`)}
        </p>,
      );
      i++;
      continue;
    }
    if (LIST.test(line)) {
      const items: string[] = [];
      const ordered = /^\s*\d+\./.test(line);
      while (i < lines.length && LIST.test(at(i))) {
        let t = at(i).replace(LIST, '');
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(at(i)) && !LIST.test(at(i))) t += ` ${at(i++).trim()}`;
        items.push(t);
      }
      const List = ordered ? 'ol' : 'ul';
      blocks.push(
        <List key={k++} {...stylex.props(text.bodySm, s.mdList)}>
          {items.map((t, j) => (
            <li key={j}>{itemLine(t, `l${k}-${j}`)}</li>
          ))}
        </List>,
      );
      continue;
    }
    if (!line.trim()) {
      i++;
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && at(i).trim() && !/^(#{1,6}\s|```)/.test(at(i)) && !LIST.test(at(i))) para.push(at(i++).trim());
    blocks.push(
      <p key={k++} {...stylex.props(text.bodySm, s.mdP)}>
        {inline(para.join(' '), `p${k}`)}
      </p>,
    );
  }
  return blocks;
}

export function Markdown({ md }: { md: string }) {
  return <div {...stylex.props(s.md)}>{mdBlocks(md).map((b, j) => <Fragment key={j}>{b}</Fragment>)}</div>;
}
