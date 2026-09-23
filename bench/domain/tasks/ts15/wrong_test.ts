// Rejects a timeout with a generic Error (name 'Error'), not a TimeoutError.
export function withTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
  options: { signal?: AbortSignal } = {},
): Promise<T> {
  const outer = options.signal;
  if (outer?.aborted) return Promise.reject(outer.reason);
  const ctl = new AbortController();
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      const e = new Error('timed out');
      ctl.abort(e);
      reject(e);
    }, ms);
    outer?.addEventListener('abort', () => { clearTimeout(timer); ctl.abort(outer.reason); reject(outer.reason); }, { once: true });
    task(ctl.signal).then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}
