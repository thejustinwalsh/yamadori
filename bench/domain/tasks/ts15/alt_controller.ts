// Hand-rolled: an AbortController, a cleared setTimeout, a DOMException for the timeout,
// and Promise.race against an abort promise.
export async function withTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
  options?: { signal?: AbortSignal },
): Promise<T> {
  const external = options?.signal;
  external?.throwIfAborted();

  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new DOMException('operation timed out', 'TimeoutError')), ms);
  const forward = () => ctl.abort(external!.reason);
  external?.addEventListener('abort', forward, { once: true });

  let stop!: () => void;
  const aborted = new Promise<never>((_, reject) => {
    const onAbort = () => reject(ctl.signal.reason);
    ctl.signal.addEventListener('abort', onAbort, { once: true });
    stop = () => ctl.signal.removeEventListener('abort', onAbort);
  });

  try {
    return await Promise.race([Promise.resolve().then(() => task(ctl.signal)), aborted]);
  } finally {
    clearTimeout(timer);
    stop();
    external?.removeEventListener('abort', forward);
  }
}
