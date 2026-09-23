// No DisposableStack: an explicit array, a reverse loop, and try/catch bookkeeping.
export class Resource {
  private closed = false;
  constructor(public readonly name: string, private readonly log: string[]) {
    this.log.push('open:' + name);
  }
  get disposed() {
    return this.closed;
  }
  [Symbol.dispose]() {
    if (!this.closed) {
      this.closed = true;
      this.log.push('close:' + this.name);
    }
  }
}

function disposeReverse(items: Disposable[]): void {
  let failed = false;
  let firstError: unknown;
  while (items.length > 0) {
    const item = items.pop()!;
    try {
      item[Symbol.dispose]();
    } catch (e) {
      if (!failed) { failed = true; firstError = e; }
    }
  }
  if (failed) throw firstError;
}

export function openAll(names: readonly string[], open: (name: string) => Disposable): Disposable {
  const opened: Disposable[] = [];
  try {
    for (const n of names) opened.push(open(n));
  } catch (e) {
    try { disposeReverse(opened); } catch { /* the open error wins */ }
    throw e;
  }
  return {
    [Symbol.dispose]() {
      disposeReverse(opened);
    },
  };
}
