// Parts of the package-onboarding surface (docs/PACKAGE-ONBOARDING.md 8.2)
// that the CREATE view's watching list and the onboarding page share: the
// stage rail drawn from the job rows, the blockers with their remedies, the
// clarify forms, the child skills, and the ONBOARDING CARD itself.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useState } from 'react';
import {
  clarifyNeeds,
  DATASET_POST,
  erroredJobOf,
  onboardingPath,
  onboardingTone,
  packageLabel,
  RAIL_TONE,
  railCells,
  rerunnableJobs,
  waitingLine,
  type Blocker,
  type OnboardingDetail,
  type OnboardingJob,
  type OnboardingSkill,
} from '../../api/onboarding';
import { statusTone } from '../../api/skills';
import { usePoll } from '../../api/usePoll';
import { ago } from '../../format';
import { href } from '../../router';
import { colors, space } from '../../tokens/tokens.stylex';
import { NavLink } from '../../ui/NavLink';
import { Chip, Label, layout, Row } from '../../ui/primitives';
import { PollState } from '../../ui/StateView';
import { text } from '../../ui/text';
import { Btn, Field, Note, s as p, send, type Msg } from './parts';

export const o = stylex.create({
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
  fields: { display: 'flex', flexWrap: 'wrap', gap: space.spaceSm, alignItems: 'flex-end' },
  blocker: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0, flexGrow: 1, textTransform: 'none' },
  join: { borderTopStyle: 'dotted' },
});

const tone = stylex.create({
  moss: { borderTopColor: colors.primaryContainer },
  cyan: { borderTopColor: colors.tertiaryContainer, boxShadow: `inset 0 12px 16px -12px color-mix(in srgb, ${colors.tertiaryContainer} 35%, transparent)` },
  crimson: { borderTopColor: colors.secondaryContainer, boxShadow: `inset 0 12px 16px -12px color-mix(in srgb, ${colors.secondaryContainer} 45%, transparent)` },
  rose: { borderTopColor: colors.secondary },
  muted: {},
});

const RAIL_WORD: Record<string, string> = { pending: 'waits', queued: 'queued', waiting: 'deferred', held: 'needs you' };

/**
 * The dataset's stages, each from its newest job row; the JOINS (no job)
 * and the jobless stages from where the dataset stands (api/onboarding
 * railCells). A JOIN's cell is drawn dotted.
 */
export function OnboardingRail({ d, now }: { d: Pick<OnboardingDetail, 'stages' | 'joins' | 'stage' | 'job_rows' | 'blockers' | 'held'>; now: number }) {
  const cells = railCells(d, now);
  return (
    <div role="list" aria-label="onboarding stages" {...stylex.props(p.rail)}>
      {cells.map((c, i) => {
        const t = RAIL_TONE[c.state];
        const title = c.job ? `${c.job.queue} ${c.job.id} · ${c.job.state}${c.job.attempts !== null ? ` · try ${c.job.attempts}/${c.job.max_attempts ?? '—'}` : ''}${c.job.progress ? ` · ${c.job.progress}` : ''}` : c.join ? 'a JOIN: no job; it waits for its blockers' : undefined;
        return (
          <div key={c.stage} role="listitem" title={title} {...stylex.props(p.cell, t && tone[t], c.join && o.join)}>
            <Label>{String(i + 1).padStart(2, '0')}</Label>
            <span {...stylex.props(text.labelXs, c.state === 'pending' ? text.outline : text.onSurface)}>{c.stage}</span>
            <span {...stylex.props(text.labelXs, t === 'crimson' || t === 'rose' ? text.rose : t === 'cyan' ? text.cyan : t === 'moss' ? text.moss : text.outline)}>{RAIL_WORD[c.state] ?? c.state}</span>
            {c.join ? <span {...stylex.props(text.labelXs, text.outline)}>join</span> : null}
          </div>
        );
      })}
    </div>
  );
}

/** Each blocker with its remedy and its owner, and a re-run where it names an errored job. */
export function Blockers({ blockers, busy, onRerun }: { blockers: Blocker[]; busy: boolean; onRerun: (job: string) => void }) {
  if (!blockers.length) return null;
  return (
    <div {...stylex.props(layout.stack)}>
      <Label>BLOCKED BY · {blockers.length}</Label>
      {blockers.map((b, i) => {
        const job = erroredJobOf(b);
        return (
          <Row key={i} bad={b.owner === 'operator' || !!b.field}>
            <div {...stylex.props(o.blocker)}>
              <span {...stylex.props(text.bodySm, b.owner === 'operator' || b.field ? p.err : p.why)}>{b.label ? `${b.label}: ` : ''}{b.what}</span>
              {b.why ? <span {...stylex.props(text.bodySm, p.why)}>{b.why}</span> : null}
              {b.remedy ? (
                <span {...stylex.props(text.bodySm, p.why)}>
                  REMEDY{b.owner ? ` (${b.owner})` : ''} {b.remedy}
                </span>
              ) : null}
              <span {...stylex.props(text.labelXs, text.outline)}>
                owner {b.owner ?? (b.field ? 'operator' : '—')}
                {b.retryable === true || b.retryable === false ? ` · retryable: ${b.retryable ? 'yes' : 'no'}` : ''}
              </span>
            </div>
            {job ? (
              <Btn disabled={busy} onClick={() => onRerun(job)} title={`POST ${DATASET_POST.rerun} {job: ${job}}`}>
                RE-RUN JOB
              </Btn>
            ) : null}
          </Row>
        );
      })}
    </div>
  );
}

/** Errored stage jobs no blocker already offers (a stage left behind by a re-run, say). */
export function ErroredJobs({ rows, blockers, busy, onRerun }: { rows: OnboardingJob[]; blockers: Blocker[]; busy: boolean; onRerun: (job: string) => void }) {
  const offered = new Set(blockers.map(erroredJobOf).filter(Boolean));
  const jobs = rerunnableJobs(rows).filter((j) => !offered.has(j.id));
  if (!jobs.length) return null;
  return (
    <div {...stylex.props(layout.stack)}>
      {jobs.map((j) => (
        <Row key={j.id} bad>
          <span {...stylex.props(p.err)}>
            {j.stage ?? j.queue}: {j.queue} {j.id} errored{j.error ? ` · ${j.error}` : ''}
          </span>
          <Btn disabled={busy} onClick={() => onRerun(j.id)}>{`RE-RUN ${j.stage ?? 'JOB'}`}</Btn>
        </Row>
      ))}
    </div>
  );
}

/**
 * Clarify: the operator's licence statement when the licence is unanswered
 * (POST /dash/api/dataset/answer {id, licence}), or the advance a person makes
 * when a found licence carries a restriction (POST /dash/api/dataset/advance).
 */
export function ClarifyForms({ d, busy, act }: { d: OnboardingDetail; busy: boolean; act: (path: string, body: unknown, ok: string) => Promise<boolean> }) {
  const need = clarifyNeeds(d);
  const [licence, setLicence] = useState('');
  if (!need.licence && !need.advance) return null;
  return (
    <div {...stylex.props(layout.stackSm)}>
      {need.licence ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void act(DATASET_POST.answer, { id: d.id, licence: licence.trim() }, 'licence recorded as yours; the onboarding moves on when nothing else blocks it').then((ok) => {
              if (ok) setLicence('');
            });
          }}
          {...stylex.props(layout.stackSm)}
        >
          <p {...stylex.props(text.bodySm, p.why)}>{need.licence.why ?? need.licence.what}</p>
          <div {...stylex.props(o.fields)}>
            <Field label="licence (your statement: SPDX id, or the terms)">
              <input aria-label="licence" value={licence} onChange={(e) => setLicence(e.target.value)} placeholder={d.licence_value ?? 'MIT'} {...stylex.props(text.bodySm, p.input)} />
            </Field>
            <Btn type="submit" disabled={busy || !licence.trim()}>
              RECORD LICENCE
            </Btn>
          </div>
        </form>
      ) : null}
      {need.advance ? (
        <div {...stylex.props(layout.stack)}>
          <p {...stylex.props(text.bodySm, p.err)}>{need.advance}</p>
          <div {...stylex.props(layout.rowWrap)}>
            <Btn danger disabled={busy} onClick={() => void act(DATASET_POST.advance, { id: d.id }, 'advanced past clarify by you')}>
              ADVANCE · SERVE IT
            </Btn>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function ChildSkills({ skills, counts }: { skills: OnboardingSkill[]; counts?: Record<string, number> }) {
  if (!skills.length) return null;
  const summary = Object.entries(counts ?? {})
    .filter(([, v]) => v)
    .map(([k, v]) => `${k} ${v}`)
    .join(' · ');
  return (
    <div {...stylex.props(layout.stack)}>
      <Label>
        SKILLS · {skills.length}
        {summary ? ` · ${summary}` : ''}
      </Label>
      {skills.map((k) => (
        <Row key={k.id}>
          <NavLink to={href.skill(k.id)} {...stylex.props(p.link)}>
            {k.name}
          </NavLink>
          <span {...stylex.props(layout.rowWrap)}>
            {k.lead_for ? <Chip tone="cyan">{`lead · ${k.lead_for}`}</Chip> : null}
            <Chip tone={statusTone(k.status)}>{k.status}</Chip>
          </span>
        </Row>
      ))}
    </div>
  );
}

/** POST, say what happened, and reload the detail on success. */
export function useAct(reload: () => void): { busy: boolean; msg: Msg; act: (path: string, body: unknown, ok: string) => Promise<boolean> } {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const act = async (path: string, body: unknown, ok: string) => {
    setBusy(true);
    const r = await send(path, body, ok);
    setBusy(false);
    setMsg(r.msg);
    if (r.data) reload();
    return !!r.data;
  };
  return { busy, msg, act };
}

/** One onboarding, polled while it walks its stages (the PipelineCard idiom). */
export function OnboardingCard({ id, onDrop }: { id: string; onDrop?: () => void }) {
  const [every, setEvery] = useState(3000);
  const d = usePoll<{ onboarding: OnboardingDetail }>(onboardingPath(id), every);
  const { busy, msg, act } = useAct(d.reload);
  const x = d.data?.onboarding;
  useEffect(() => {
    if (x) setEvery(x.stage === 'complete' ? 20000 : 3000);
  }, [x?.stage]);
  if (!x) {
    return (
      <div {...stylex.props(o.card)}>
        <PollState path={onboardingPath(id)} failure={d.failure} />
        {d.failure && onDrop ? <Btn onClick={onDrop}>STOP WATCHING</Btn> : null}
      </div>
    );
  }
  const now = Date.now() / 1000;
  const waiting = waitingLine(x);
  const rerun = (job: string) => void act(DATASET_POST.rerun, { job }, `job ${job} re-queued`);
  return (
    <div {...stylex.props(o.card)} aria-label={`onboarding ${packageLabel(x)}`}>
      <div {...stylex.props(layout.between)}>
        <NavLink to={href.onboarding(x.id)} {...stylex.props(text.labelMd, p.link, p.asIs)}>
          {packageLabel(x)}
        </NavLink>
        <span {...stylex.props(layout.rowWrap)}>
          <Chip tone={onboardingTone(x.state)}>{x.state}</Chip>
          <Chip tone="muted">{x.stage}</Chip>
          {x.group ? <Chip tone="muted">{`group ${x.group}`}</Chip> : null}
        </span>
      </div>
      <OnboardingRail d={x} now={now} />
      {waiting ? <p {...stylex.props(text.bodySm, p.why)}>WAITING · {waiting}</p> : null}
      <Blockers blockers={x.blockers} busy={busy} onRerun={rerun} />
      <ErroredJobs rows={x.job_rows} blockers={x.blockers} busy={busy} onRerun={rerun} />
      <ClarifyForms d={x} busy={busy} act={act} />
      <ChildSkills skills={x.skills} counts={x.skill_counts} />
      <div {...stylex.props(layout.rowWrap)}>
        <NavLink to={href.onboarding(x.id)} {...stylex.props(text.labelXs, p.link)}>
          EVERY STAGE'S RESULT →
        </NavLink>
        {onDrop ? <Btn onClick={onDrop}>STOP WATCHING</Btn> : null}
        <span {...stylex.props(text.labelXs, text.outline)}>updated {ago(now - x.updated)} ago</span>
      </div>
      <Note msg={msg} />
    </div>
  );
}
