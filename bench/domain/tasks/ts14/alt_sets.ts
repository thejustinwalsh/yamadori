// Registrations as unique token objects in a Set per event (Sets iterate in insertion order),
// with the constraint written as a conditional on keyof.
type Args<E, K extends keyof E> = E[K] extends unknown[] ? E[K] : never;

export class TypedEmitter<E extends Record<keyof E, unknown[]>> {
  private readonly regs: { [K in keyof E]?: Set<{ call: (...a: Args<E, K>) => void; single: boolean }> } = {};

  on<K extends keyof E>(name: K, cb: (...a: Args<E, K>) => void): () => void {
    return this.register(name, cb, false);
  }

  once<K extends keyof E>(name: K, cb: (...a: Args<E, K>) => void): () => void {
    return this.register(name, cb, true);
  }

  emit<K extends keyof E>(name: K, ...a: Args<E, K>): boolean {
    const set = this.regs[name];
    if (!set || set.size === 0) return false;
    const snapshot = Array.from(set);
    for (const reg of snapshot) {
      if (reg.single) set.delete(reg);
      reg.call(...a);
    }
    return true;
  }

  listenerCount<K extends keyof E>(name: K): number {
    return this.regs[name]?.size ?? 0;
  }

  private register<K extends keyof E>(name: K, call: (...a: Args<E, K>) => void, single: boolean): () => void {
    const token = { call, single };
    const set = (this.regs[name] ??= new Set());
    set.add(token);
    return () => void set.delete(token);
  }
}
