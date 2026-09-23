type Listener<A extends unknown[]> = (...args: A) => void;

export class TypedEmitter<Events extends { [K in keyof Events]: unknown[] }> {
  #listeners = new Map<keyof Events, { fn: Listener<any>; once: boolean }[]>();

  on<K extends keyof Events>(event: K, listener: Listener<Events[K]>): () => void {
    return this.#add(event, listener, false);
  }

  once<K extends keyof Events>(event: K, listener: Listener<Events[K]>): () => void {
    return this.#add(event, listener, true);
  }

  emit<K extends keyof Events>(event: K, ...args: Events[K]): boolean {
    const list = this.#listeners.get(event);
    if (!list || list.length === 0) return false;
    for (const entry of [...list]) {
      if (entry.once) this.#remove(event, entry);
      entry.fn(...args);
    }
    return true;
  }

  listenerCount(event: keyof Events): number {
    return this.#listeners.get(event)?.length ?? 0;
  }

  #add<K extends keyof Events>(event: K, fn: Listener<Events[K]>, once: boolean): () => void {
    const entry = { fn, once };
    const list = this.#listeners.get(event) ?? [];
    list.push(entry);
    this.#listeners.set(event, list);
    return () => this.#remove(event, entry);
  }

  #remove(event: keyof Events, entry: { fn: Listener<any>; once: boolean }): void {
    const list = this.#listeners.get(event);
    const i = list ? list.indexOf(entry) : -1;
    if (list && i >= 0) list.splice(i, 1);
  }
}
