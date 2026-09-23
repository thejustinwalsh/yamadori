// Unsubscribes by function identity: registering the same function twice, one `off` removes both.
export class TypedEmitter<Events extends { [K in keyof Events]: unknown[] }> {
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
      if (entry.once) this.listeners.set(event, (this.listeners.get(event) ?? []).filter((e) => e !== entry));
      entry.fn(...args);
    }
    return true;
  }
  listenerCount(event: keyof Events): number {
    return this.listeners.get(event)?.length ?? 0;
  }
  private add<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void, once: boolean): () => void {
    const list = this.listeners.get(event) ?? [];
    list.push({ fn: fn as (...args: any[]) => void, once });
    this.listeners.set(event, list);
    return () => {
      this.listeners.set(event, (this.listeners.get(event) ?? []).filter((e) => e.fn !== fn));
    };
  }
}
