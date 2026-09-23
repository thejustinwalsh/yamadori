// The rules the ADD DATASET flow applies in the browser. Each one is small,
// and each is here rather than inline so vitest can hold it still: which
// submission shape a paste becomes, which badge a field wears, and when a
// card is worth polling fast.
//
// What is NOT here: whether a field is answered, and where its value came
// from. That is datasets.field_states() in Python, so the classic page, the
// worker and this page cannot disagree about it.
import type { Tone } from '../ui/primitives';
import type { Failure, Refusal } from './client';
import type { Dataset, FieldState, JobRow, Provenance } from './types';

export type Submission = { url: string; kind: string } | { text: string; kind: string };

/**
 * A paste is a URL when it is ONE http(s) URL and nothing else; anything else
 * is text. Mirrors the server's own rule in datasets.create(), so the page
 * never sends {text} for something the server would have fetched.
 */
export function submissionBody(input: string, kind: string): Submission | { error: string } {
  const v = input.trim();
  if (!v) return { error: 'nothing to submit: paste a URL or the text itself' };
  if (/^https?:\/\/\S+$/.test(v)) return { url: v, kind };
  return { text: v, kind };
}

export type Badge = { label: string; tone: Tone; title: string };

/** One badge per provenance. The word carries the meaning; colour doubles it. */
export const BADGES: Record<Provenance, Badge> = {
  evidence: {
    label: 'EVIDENCE',
    tone: 'moss',
    title: 'Filled from a verbatim quote the worker found in the fetched text and checked character for character.',
  },
  proposed: {
    label: 'PROPOSED',
    tone: 'cyan',
    title: "The model's reading of the source. An inference, not a quote: override it if it is wrong.",
  },
  operator: {
    label: 'OPERATOR',
    tone: 'muted',
    title: 'Typed by a person, at submission or since.',
  },
  needs_you: {
    label: 'NEEDS YOU',
    tone: 'crimson',
    title: 'Not established from the source. It blocks clarify until a person answers it.',
  },
};

export function badgeFor(state: string): Badge {
  return (BADGES as Record<string, Badge>)[state] ?? BADGES.needs_you;
}

/** A value as the card prints it. Empty is said in words, never drawn blank. */
export function displayValue(v: FieldState['value']): string {
  if (v === null || v === undefined) return 'not answered';
  if (Array.isArray(v)) return v.length ? v.join(', ') : 'not answered';
  return v.trim() ? v : 'not answered';
}

/** Only http(s) locations become links; "the pasted source text" stays text. */
export function linkable(where: string | null | undefined): where is string {
  return !!where && /^https?:\/\//.test(where);
}

/** The body POST /dash/api/dataset/answer takes for one field. */
export function answerBody(datasetId: string, f: FieldState, draft: string | string[]): Record<string, unknown> {
  const value = f.input === 'multi' ? (Array.isArray(draft) ? draft : draft.split(',').map((x) => x.trim()).filter(Boolean)) : String(draft).trim();
  return { id: datasetId, [f.field]: value };
}

/** The draft an edit starts from: the current value, never a suggestion. */
export function draftOf(f: FieldState): string | string[] {
  if (f.input === 'multi') return Array.isArray(f.value) ? [...f.value] : [];
  return typeof f.value === 'string' ? f.value : '';
}

const LIVE = new Set(['queued', 'running']);

/** Jobs that are still going to change something. */
export function liveJobs(d: Pick<Dataset, 'job_rows'>): JobRow[] {
  return (d.job_rows ?? []).filter((j) => LIVE.has(j.state));
}

/** Poll fast while a job is queued or running, slowly otherwise. */
export function pollEvery(d: Pick<Dataset, 'job_rows'> | null): number {
  if (!d) return 3000;
  return liveJobs(d).length ? 3000 : 15000;
}

/** "assist · running · looking for a licence file: …" -- verbatim, no bar. */
export function jobLine(j: JobRow): string {
  const what = j.queue.replace(/^dataset\./, '');
  const tail = j.state === 'errored' ? j.error : j.progress;
  return [what, j.state, tail].filter(Boolean).join(' · ');
}

/** The headline a card prints for what the dataset is waiting on. */
export function waitingOn(d: Dataset): string {
  const live = liveJobs(d);
  if (live.length) {
    const j = live[0] as JobRow;
    return `${j.queue.replace(/^dataset\./, '')} ${j.state}${j.progress ? `: ${j.progress}` : ''}`;
  }
  if (d.stage === 'complete') return 'complete';
  if ((d.job_rows ?? d.errored_jobs ?? []).some((j) => j.state === 'errored')) return 'a job errored';
  const needs = (d.field_states ?? []).filter((f) => f.state === 'needs_you').length;
  if (d.stage === 'clarify' && needs) return `${needs} field${needs === 1 ? '' : 's'} need${needs === 1 ? 's' : ''} you`;
  if (d.stage === 'clarify' && d.assist?.held) return 'held: a person decides';
  if (d.stage === 'clarify') return 'every field answered; advance it';
  return d.stage;
}

/** A refused or failed POST, as one line: the server's error, then each reason. */
export function refusalText(r: { failure: Failure; reasons: Refusal[] }): string {
  const f = r.failure;
  const head =
    f.kind === 'nokey' ? 'no key in this browser' : f.kind === 'http' ? `HTTP ${f.status}: ${f.message}` : f.message;
  const why = r.reasons.map((x) => x.what).filter((x): x is string => !!x);
  return why.length ? `${head} -- ${why.join('; ')}` : head;
}
