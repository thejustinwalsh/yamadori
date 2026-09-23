// A single waiter slot: a second pending next() overwrites the first, which then never settles.
export class AsyncQueue<T> implements AsyncIterable<T> {
  #items: T[] = [];
  #waiter: ((r: IteratorResult<T, undefined>) => void) | null = null;
  #closed = false;
  get size(): number { return this.#items.length; }
  get closed(): boolean { return this.#closed; }
  push(item: T): void {
    if (this.#closed) throw new Error('queue is closed');
    const w = this.#waiter;
    this.#waiter = null;
    if (w) w({ done: false, value: item });
    else this.#items.push(item);
  }
  close(): void {
    this.#closed = true;
    this.#waiter?.({ done: true, value: undefined });
    this.#waiter = null;
  }
  [Symbol.asyncIterator](): AsyncIterator<T, undefined> {
    return {
      next: () => {
        if (this.#items.length > 0) return Promise.resolve({ done: false, value: this.#items.shift()! });
        if (this.#closed) return Promise.resolve({ done: true, value: undefined });
        return new Promise((resolve) => { this.#waiter = resolve; });
      },
    };
  }
}
