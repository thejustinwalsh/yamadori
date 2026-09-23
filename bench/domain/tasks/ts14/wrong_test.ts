// Iterates the live listener array while `once` splices it: the listener after a once-listener is skipped.
export class TypedEmitter<Events extends { [K in keyof Events]: unknown[] }> {
  private listeners = new Map<keyof Events, { fn: (...args: any[]) => void; once: boolean }[]>();

  on<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): () => void {
    return this.add(event, fn, false);
  }
  once<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): () => void {
    return this.add(event, fn, true);
  }
  emit<K extends keyof Events>(event: K, ...args: Events[K]): boolean {
    const list = this.listeners.get(event) ?? [];
    list.forEach((entry, i) => {
      if (entry.once) list.splice(i, 1);
      entry.fn(...args);
    });
    return list.length > 0 || false;
  }
  listenerCount(event: keyof Events): number {
    return this.listeners.get(event)?.length ?? 0;
  }
  private add<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void, once: boolean): () => void {
    const entry = { fn: fn as (...args: any[]) => void, once };
    const list = this.listeners.get(event) ?? [];
    list.push(entry);
    this.listeners.set(event, list);
    return () => {
      const i = list.indexOf(entry);
      if (i >= 0) list.splice(i, 1);
    };
  }
}
