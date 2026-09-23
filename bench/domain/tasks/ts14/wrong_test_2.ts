// Constrains Events to Record<string, unknown[]>: an interface has no index signature, so it is rejected.
export class TypedEmitter<Events extends Record<string, unknown[]>> {
  private listeners = new Map<keyof Events, { fn: (...args: any[]) => void; once: boolean }[]>();

  on<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): () => void {
    return this.add(event, fn, false);
  }
  once<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): () => void {
    return this.add(event, fn, true);
  }
  emit<K extends keyof Events>(event: K, ...args: Events[K]): boolean {
    const list = this.listeners.get(event);
    if (!list?.length) return false;
    for (const entry of [...list]) {
      if (entry.once) list.splice(list.indexOf(entry), 1);
      entry.fn(...args);
    }
    return true;
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
