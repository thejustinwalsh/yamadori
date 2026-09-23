// Polling with the discipline the Python vitals page established: the next
// poll is scheduled only after the last one finished, a snapshot older than
// STALE_AFTER is marked stale rather than shown as current, and a 401 drops
// every value -- a stale number under a sign-in prompt is a number nobody is
// standing behind.
//
// Every poll reads and writes the shared snapshot cache (./cache.ts). A
// screen reached from a nav link therefore mounts with the snapshot the
// link's hover/press already fetched, or joins that request in flight, and
// is polled from the snapshot's age rather than fetched a second time.
import { useEffect, useState, useSyncExternalStore } from 'react';
import { fetchShared, invalidate, isInflight, peek } from './cache';
import { readKey, type Failure } from './client';

export const STALE_AFTER_S = 30; // mcp/dash_vitals.py STALE_AFTER

export type Poll<T> = {
  data: T | null;
  receivedAt: number | null;
  failure: Failure | null;
  inflight: boolean;
  /** seconds since the last good snapshot, ticking */
  age: number | null;
  stale: boolean;
  reload: () => void;
};

function subscribeKey(cb: () => void) {
  window.addEventListener('yamadori:key', cb);
  window.addEventListener('storage', cb);
  return () => {
    window.removeEventListener('yamadori:key', cb);
    window.removeEventListener('storage', cb);
  };
}

export function useKey(): string {
  return useSyncExternalStore(subscribeKey, readKey, () => '');
}

export function useNow(everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), everyMs);
    return () => window.clearInterval(id);
  }, [everyMs]);
  return now;
}

type Held<T> = { path: string | null; data: T | null; receivedAt: number | null; failure: Failure | null };

function seed<T>(path: string | null): Held<T> {
  const s = path ? peek<T>(path) : null;
  return { path, data: s ? s.data : null, receivedAt: s ? s.receivedAt : null, failure: null };
}

export function usePoll<T>(path: string | null, everyMs: number): Poll<T> {
  const key = useKey();
  const [held, setHeld] = useState<Held<T>>(() => seed<T>(path));
  const [inflight, setInflight] = useState(false);
  const [nonce, setNonce] = useState(0);
  const now = useNow(1000);
  // A new path (another dataset id) starts from its own snapshot, never from
  // the previous path's numbers.
  const cur = held.path === path ? held : seed<T>(path);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    let timer: number | null = null;
    const run = async () => {
      setInflight(true);
      const r = await fetchShared<T>(path);
      if (!alive) return; // unmounted, or the path/key changed
      setInflight(false);
      if (r.ok) {
        setHeld({ path, data: r.data, receivedAt: r.receivedAt, failure: null });
      } else if (r.failure.kind === 'unauthorised' || r.failure.kind === 'nokey') {
        setHeld({ path, data: null, receivedAt: null, failure: r.failure });
        return; // no retry until the key changes
      } else {
        const f = r.failure;
        setHeld((h) => ({ ...(h.path === path ? h : seed<T>(path)), failure: f }));
      }
      if (everyMs > 0) timer = window.setTimeout(run, everyMs);
    };
    const s = peek<T>(path);
    const age = s ? Date.now() - s.receivedAt : Infinity;
    if (s && everyMs > 0 && age < everyMs && !isInflight(path)) {
      setHeld({ path, data: s.data, receivedAt: s.receivedAt, failure: null });
      timer = window.setTimeout(run, everyMs - age);
    } else {
      void run();
    }
    return () => {
      alive = false;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [path, everyMs, key, nonce]);

  const age = cur.receivedAt === null ? null : Math.max(0, Math.round((now - cur.receivedAt) / 1000));
  return {
    data: cur.data,
    receivedAt: cur.receivedAt,
    failure: cur.failure,
    inflight,
    age,
    stale: age !== null && age > STALE_AFTER_S,
    reload: () => {
      if (path) invalidate(path);
      setNonce((n) => n + 1);
    },
  };
}
