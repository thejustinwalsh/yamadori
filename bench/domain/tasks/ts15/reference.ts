export function withTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
  options: { signal?: AbortSignal } = {},
): Promise<T> {
  const outer = options.signal;
  if (outer?.aborted) return Promise.reject(outer.reason);
  const signal = outer ? AbortSignal.any([outer, AbortSignal.timeout(ms)]) : AbortSignal.timeout(ms);

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
    const done = () => signal.removeEventListener('abort', onAbort);
    let pending: Promise<T>;
    try {
      pending = task(signal);
    } catch (e) {
      done();
      reject(e);
      return;
    }
    Promise.resolve(pending).then(
      (v) => { done(); resolve(v); },
      (e) => { done(); reject(e); },
    );
  });
}
