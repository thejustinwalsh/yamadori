// close() ends iteration immediately and drops items that were pushed but not yet consumed.
export class AsyncQueue<T> implements AsyncIterable<T> {
  #items: T[] = [];
  #waiters: ((r: IteratorResult<T, undefined>) => void)[] = [];
  #closed = false;
  get size(): number { return this.#items.length; }
  get closed(): boolean { return this.#closed; }
  push(item: T): void {
    if (this.#closed) throw new Error('queue is closed');
    const w = this.#waiters.shift();
    if (w) w({ done: false, value: item });
    else this.#items.push(item);
  }
  close(): void {
    this.#closed = true;
    this.#items.length = 0;
    for (const w of this.#waiters.splice(0)) w({ done: true, value: undefined });
  }
  [Symbol.asyncIterator](): AsyncIterator<T, undefined> {
    return {
      next: () => {
        if (this.#items.length > 0) return Promise.resolve({ done: false, value: this.#items.shift()! });
        if (this.#closed) return Promise.resolve({ done: true, value: undefined });
        return new Promise((resolve) => this.#waiters.push(resolve));
      },
    };
  }
}
