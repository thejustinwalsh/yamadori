// One snapshot per /dash/api path, shared by every poll and by route intent.
//
// Hovering, focusing or pressing a nav link starts the fetch for the screen
// it points at (src/routes.ts); the screen's usePoll then starts from the
// snapshot already here, or joins the request still in flight, instead of
// mounting empty and fetching again. Polls write back, so two screens that
// read the same path (NEBARI and SENTEI both read /results) share it.
//
// Only good snapshots are kept. A 401 or a missing key drops everything: a
// cached number under a sign-in prompt is a number nobody is standing behind.
import { getJson, type Fetched } from './client';

export type Snapshot<T> = { data: T; receivedAt: number };

const MAX_ENTRIES = 32;
const snapshots = new Map<string, Snapshot<unknown>>();
const inflight = new Map<string, Promise<Fetched<unknown>>>();

export function peek<T>(path: string): Snapshot<T> | null {
  return (snapshots.get(path) as Snapshot<T> | undefined) ?? null;
}

export function isFresh(path: string, maxAgeMs: number): boolean {
  const s = snapshots.get(path);
  return !!s && Date.now() - s.receivedAt < maxAgeMs;
}

export function isInflight(path: string): boolean {
  return inflight.has(path);
}

function put(path: string, snap: Snapshot<unknown>): void {
  snapshots.delete(path); // re-insert: Map order is the LRU order
  snapshots.set(path, snap);
  while (snapshots.size > MAX_ENTRIES) {
    const oldest = snapshots.keys().next().value;
    if (oldest === undefined) break;
    snapshots.delete(oldest);
  }
}

export function invalidate(path: string): void {
  snapshots.delete(path);
}

export function clearAll(): void {
  snapshots.clear();
}

/**
 * GET `path`, sharing one request between every caller that asks while it is
 * in flight. Never rejects: a thrown fetch becomes a `network` failure.
 */
export function fetchShared<T>(path: string): Promise<Fetched<T>> {
  const running = inflight.get(path);
  if (running) return running as Promise<Fetched<T>>;
  const p = getJson<T>(path)
    .catch((e: unknown): Fetched<T> => ({ ok: false, failure: { kind: 'network', message: `cannot reach ${path}: ${String(e)}` } }))
    .then((r) => {
      inflight.delete(path);
      if (r.ok) put(path, { data: r.data, receivedAt: r.receivedAt });
      else if (r.failure.kind === 'unauthorised' || r.failure.kind === 'nokey') clearAll();
      return r;
    });
  inflight.set(path, p as Promise<Fetched<unknown>>);
  return p;
}

/** Warm `path` unless a snapshot younger than `maxAgeMs` is already here. */
export function prefetch(path: string, maxAgeMs = 10_000): Promise<void> {
  if (isFresh(path, maxAgeMs)) return Promise.resolve();
  return fetchShared(path).then(() => undefined);
}

if (typeof window !== 'undefined') {
  // A different key may see different data, or none.
  window.addEventListener('yamadori:key', clearAll);
  window.addEventListener('storage', (e) => {
    if (e.key === null || e.key === 'yamadori_key') clearAll();
  });
}
