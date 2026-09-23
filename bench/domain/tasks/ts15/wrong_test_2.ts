// Waits for the task to notice the abort: a task that ignores its signal never times out.
export async function withTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
  options: { signal?: AbortSignal } = {},
): Promise<T> {
  const signals = [AbortSignal.timeout(ms)];
  if (options.signal) signals.push(options.signal);
  const signal = AbortSignal.any(signals);
  signal.throwIfAborted();
  return await task(signal);
}
