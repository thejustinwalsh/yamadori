// Class-based, with Promise.withResolvers for the waiters and a class for the lease.
class PoolLease<T> implements AsyncDisposable {
  #done = false;
  constructor(readonly value: T, private readonly giveBack: (v: T) => void) {}
  async [Symbol.asyncDispose](): Promise<void> {
    if (this.#done) return;
    this.#done = true;
    await Promise.resolve();
    this.giveBack(this.value);
  }
}

class Pool<T> {
  readonly #idle: T[];
  readonly #queue: PromiseWithResolvers<PoolLease<T>>[] = [];

  constructor(items: readonly T[]) {
    this.#idle = items.slice();
  }

  get available(): number { return this.#idle.length; }
  get waiting(): number { return this.#queue.length; }

  acquire(): Promise<PoolLease<T>> {
    if (this.#idle.length > 0) return Promise.resolve(this.#wrap(this.#idle.shift()!));
    const d = Promise.withResolvers<PoolLease<T>>();
    this.#queue.push(d);
    return d.promise;
  }

  #wrap(v: T): PoolLease<T> {
    return new PoolLease(v, (back) => {
      const next = this.#queue.shift();
      if (next) next.resolve(this.#wrap(back));
      else this.#idle.push(back);
    });
  }
}

export function createPool<T>(items: readonly T[]): Pool<T> {
  return new Pool(items);
}
