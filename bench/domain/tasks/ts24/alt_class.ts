// A hand-written iterator object (no generator), typed via a mapped tuple of element types.
type Elements<I extends readonly Iterable<unknown>[]> = { -readonly [K in keyof I]: I[K] extends Iterable<infer E> ? E : never };

class Zipped<T extends unknown[]> implements IterableIterator<T> {
  #its: Iterator<unknown>[];
  #finished = false;
  constructor(sources: readonly Iterable<unknown>[]) {
    this.#its = sources.map((s) => s[Symbol.iterator]());
    if (this.#its.length === 0) this.#finished = true;
  }
  [Symbol.iterator](): this {
    return this;
  }
  next(): IteratorResult<T, undefined> {
    if (this.#finished) return { done: true, value: undefined };
    const row: unknown[] = [];
    for (let i = 0; i < this.#its.length; i++) {
      const r = this.#its[i]!.next();
      if (r.done) {
        this.#close(i);
        return { done: true, value: undefined };
      }
      row.push(r.value);
    }
    return { done: false, value: row as T };
  }
  return(): IteratorResult<T, undefined> {
    this.#close(-1);
    return { done: true, value: undefined };
  }
  #close(exhausted: number): void {
    if (this.#finished) return;
    this.#finished = true;
    this.#its.forEach((it, i) => { if (i !== exhausted) it.return?.(); });
  }
}

export function zip<I extends Iterable<unknown>[]>(...sources: I): IterableIterator<Elements<I>> {
  return new Zipped<Elements<I>>(sources);
}
