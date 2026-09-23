// Serves waiters last-in-first-out (pop instead of shift).
export function createPool<T>(items: readonly T[]) {
  const free = [...items];
  const waiters: ((lease: { value: T } & AsyncDisposable) => void)[] = [];
  const lease = (value: T): { value: T } & AsyncDisposable => {
    let released = false;
    return {
      value,
      async [Symbol.asyncDispose]() {
        if (released) return;
        released = true;
        const next = waiters.pop();
        if (next) next(lease(value));
        else free.push(value);
      },
    };
  };
  return {
    acquire(): Promise<{ value: T } & AsyncDisposable> {
      if (free.length > 0) return Promise.resolve(lease(free.shift()!));
      return new Promise((resolve) => waiters.push(resolve));
    },
    get available() { return free.length; },
    get waiting() { return waiters.length; },
  };
}
