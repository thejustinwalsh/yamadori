export interface Lease<T> extends AsyncDisposable {
  readonly value: T;
}

export interface Pool<T> {
  acquire(): Promise<Lease<T>>;
  readonly available: number;
  readonly waiting: number;
}

export function createPool<T>(items: readonly T[]): Pool<T> {
  const free = [...items];
  const waiters: ((lease: Lease<T>) => void)[] = [];

  const lease = (value: T): Lease<T> => {
    let released = false;
    return {
      value,
      async [Symbol.asyncDispose]() {
        if (released) return;
        released = true;
        const next = waiters.shift();
        if (next) next(lease(value));
        else free.push(value);
      },
    };
  };

  return {
    acquire() {
      if (free.length > 0) return Promise.resolve(lease(free.shift()!));
      return new Promise<Lease<T>>((resolve) => waiters.push(resolve));
    },
    get available() {
      return free.length;
    },
    get waiting() {
      return waiters.length;
    },
  };
}
