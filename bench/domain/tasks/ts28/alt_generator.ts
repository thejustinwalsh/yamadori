// The iterator is an async generator (which queues concurrent next() calls itself); waiting is
// done with Promise.withResolvers.
export class AsyncQueue<T> {
  private buffer: T[] = [];
  private isClosed = false;
  private wake: PromiseWithResolvers<void> | null = null;

  get size() { return this.buffer.length; }
  get closed() { return this.isClosed; }

  push(item: T): void {
    if (this.isClosed) throw new TypeError('push after close');
    this.buffer.push(item);
    this.wake?.resolve();
  }

  close(): void {
    this.isClosed = true;
    this.wake?.resolve();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<T, void, undefined> {
    while (true) {
      if (this.buffer.length > 0) {
        yield this.buffer.shift()!;
        continue;
      }
      if (this.isClosed) return;
      this.wake = Promise.withResolvers<void>();
      await this.wake.promise;
      this.wake = null;
    }
  }
}
