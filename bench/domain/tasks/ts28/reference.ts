export class AsyncQueue<T> implements AsyncIterable<T> {
  #items: T[] = [];
  #waiters: ((r: IteratorResult<T, undefined>) => void)[] = [];
  #closed = false;

  get size(): number {
    return this.#items.length;
  }

  get closed(): boolean {
    return this.#closed;
  }

  push(item: T): void {
    if (this.#closed) throw new Error('queue is closed');
    const waiter = this.#waiters.shift();
    if (waiter) waiter({ done: false, value: item });
    else this.#items.push(item);
  }

  close(): void {
    this.#closed = true;
    for (const waiter of this.#waiters.splice(0)) waiter({ done: true, value: undefined });
  }

  next(): Promise<IteratorResult<T, undefined>> {
    if (this.#items.length > 0) return Promise.resolve({ done: false, value: this.#items.shift()! });
    if (this.#closed) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve) => this.#waiters.push(resolve));
  }

  [Symbol.asyncIterator](): AsyncIterator<T, undefined> {
    return { next: () => this.next() };
  }
}
