// Leaks on a failed open: the resources opened before the failure are never disposed.
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
  const stack = new DisposableStack();
  for (const name of names) stack.use(open(name));
  return { [Symbol.dispose]: () => stack.dispose() };
}
