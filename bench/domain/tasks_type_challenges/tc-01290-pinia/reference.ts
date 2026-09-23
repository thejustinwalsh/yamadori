type PiniaGetterValues<G> = {
  readonly [K in keyof G]: G[K] extends (...args: any[]) => infer R ? R : never
}

declare function defineStore<S, G, A>(store: {
  id: string
  state: () => S
  getters?: G & ThisType<Readonly<S> & PiniaGetterValues<G>>
  actions?: A & ThisType<S & PiniaGetterValues<G> & A>
}): S & PiniaGetterValues<G> & A
