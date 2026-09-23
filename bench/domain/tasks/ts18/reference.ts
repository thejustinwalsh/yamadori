export function memoize<This extends object, A, R>(
  method: (this: This, arg: A) => R,
  _context: ClassMethodDecoratorContext<This, (this: This, arg: A) => R>,
): (this: This, arg: A) => R {
  const caches = new WeakMap<This, Map<A, R>>();
  return function (this: This, arg: A): R {
    let cache = caches.get(this);
    if (!cache) {
      cache = new Map();
      caches.set(this, cache);
    }
    if (cache.has(arg)) return cache.get(arg)!;
    const result = method.call(this, arg);
    cache.set(arg, result);
    return result;
  };
}
