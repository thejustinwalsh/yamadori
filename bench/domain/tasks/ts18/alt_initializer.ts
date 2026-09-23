// Uses context.addInitializer to give each instance its own cache in a symbol-keyed slot,
// and types the method through a single-argument function type parameter.
type OneArg = (this: any, arg: any) => any;

export function memoize<M extends OneArg>(target: M, context: ClassMethodDecoratorContext<ThisParameterType<M>, M>): M {
  const slot = Symbol(`memo:${String(context.name)}`);
  context.addInitializer(function (this: any) {
    Object.defineProperty(this, slot, { value: new Map(), enumerable: false });
  });
  return function (this: any, arg: Parameters<M>[0]) {
    const cache: Map<unknown, ReturnType<M>> = this[slot];
    if (!cache.has(arg)) cache.set(arg, target.call(this, arg));
    return cache.get(arg);
  } as M;
}
