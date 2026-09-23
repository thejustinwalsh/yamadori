// Accepts methods of any arity (rest parameters), so misuse on a two-argument method compiles.
export function memoize<This extends object, A extends unknown[], R>(
  method: (this: This, ...args: A) => R,
  _context: ClassMethodDecoratorContext<This, (this: This, ...args: A) => R>,
): (this: This, ...args: A) => R {
  const caches = new WeakMap<This, Map<unknown, R>>();
  return function (this: This, ...args: A): R {
    let cache = caches.get(this);
    if (!cache) caches.set(this, (cache = new Map()));
    const key = args[0];
    if (!cache.has(key)) cache.set(key, method.call(this, ...args));
    return cache.get(key)!;
  };
}
