// Promise.race against a setTimeout that is never cleared: a 60 s timer keeps the process alive after success.
export function withTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
  options: { signal?: AbortSignal } = {},
): Promise<T> {
  const outer = options.signal;
  if (outer?.aborted) return Promise.reject(outer.reason);
  const ctl = new AbortController();
  outer?.addEventListener('abort', () => ctl.abort(outer.reason), { once: true });
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => ctl.abort(new DOMException('timed out', 'TimeoutError')), ms));
  const aborted = new Promise<never>((_, reject) =>
    ctl.signal.addEventListener('abort', () => reject(ctl.signal.reason), { once: true }));
  return Promise.race([Promise.resolve().then(() => task(ctl.signal)), timeout, aborted]);
}
