// Stops at the first disposal error, so the remaining resources are never disposed.
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
  const closeAll = () => { while (opened.length) opened.pop()![Symbol.dispose](); };
  try {
    for (const n of names) opened.push(open(n));
  } catch (e) {
    closeAll();
    throw e;
  }
  return { [Symbol.dispose]: closeAll };
}
