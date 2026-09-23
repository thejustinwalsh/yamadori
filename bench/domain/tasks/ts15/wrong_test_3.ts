// Ignores the caller's signal entirely: only the timeout can cancel.
export function withTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
  _options: { signal?: AbortSignal } = {},
): Promise<T> {
  const signal = AbortSignal.timeout(ms);
  return new Promise<T>((resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
    task(signal).then(resolve, reject);
  });
}
