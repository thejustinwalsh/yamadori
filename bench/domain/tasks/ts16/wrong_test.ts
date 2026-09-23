// Disposes in opening order instead of reverse order.
export class Resource implements Disposable {
  #disposed = false;
  constructor(readonly name: string, private readonly log: string[]) { log.push(`open:${name}`); }
  get disposed(): boolean { return this.#disposed; }
  [Symbol.dispose](): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.log.push(`close:${this.name}`);
  }
}
export function openAll(names: readonly string[], open: (name: string) => Disposable): Disposable {
  const opened: Disposable[] = [];
  try {
    for (const n of names) opened.push(open(n));
  } catch (e) {
    for (const r of opened) r[Symbol.dispose]();
    throw e;
  }
  let done = false;
  return {
    [Symbol.dispose]() {
      if (done) return;
      done = true;
      for (const r of opened) r[Symbol.dispose]();
    },
  };
}
