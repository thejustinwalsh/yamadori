// Uses the synchronous dispose protocol: `await using` still works, but the lease is not AsyncDisposable.
export function createPool<T>(items: readonly T[]) {
  const free = [...items];
  const waiters: ((lease: { value: T } & Disposable) => void)[] = [];
  const lease = (value: T): { value: T } & Disposable => {
    let released = false;
    return {
      value,
      [Symbol.dispose]() {
        if (released) return;
        released = true;
        const next = waiters.shift();
        if (next) next(lease(value));
        else free.push(value);
      },
    };
  };
  return {
    acquire(): Promise<{ value: T } & Disposable> {
      if (free.length > 0) return Promise.resolve(lease(free.shift()!));
      return new Promise((resolve) => waiters.push(resolve));
    },
    get available() { return free.length; },
    get waiting() { return waiters.length; },
  };
}
