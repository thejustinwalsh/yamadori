// One cache per decorated method, shared by every instance: instance B gets instance A's results.
export function memoize<This, A, R>(
  method: (this: This, arg: A) => R,
  _context: ClassMethodDecoratorContext<This, (this: This, arg: A) => R>,
): (this: This, arg: A) => R {
  const cache = new Map<A, R>();
  return function (this: This, arg: A): R {
    if (!cache.has(arg)) cache.set(arg, method.call(this, arg));
    return cache.get(arg)!;
  };
}
