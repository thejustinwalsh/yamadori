// Keys the cache by JSON.stringify(arg): distinct objects with equal contents collide, and NaN/undefined keys break.
export function memoize<This extends object, A, R>(
  method: (this: This, arg: A) => R,
  _context: ClassMethodDecoratorContext<This, (this: This, arg: A) => R>,
): (this: This, arg: A) => R {
  const caches = new WeakMap<This, Map<string, R>>();
  return function (this: This, arg: A): R {
    let cache = caches.get(this);
    if (!cache) caches.set(this, (cache = new Map()));
    const key = JSON.stringify(arg);
    if (!cache.has(key)) cache.set(key, method.call(this, arg));
    return cache.get(key)!;
  };
}
