// Legacy experimentalDecorators signature (target, key, descriptor): not callable as a standard decorator.
export function memoize(_target: object, _key: string, descriptor: PropertyDescriptor): PropertyDescriptor {
  const original = descriptor.value as (arg: unknown) => unknown;
  const caches = new WeakMap<object, Map<unknown, unknown>>();
  descriptor.value = function (this: object, arg: unknown) {
    let cache = caches.get(this);
    if (!cache) caches.set(this, (cache = new Map()));
    if (!cache.has(arg)) cache.set(arg, original.call(this, arg));
    return cache.get(arg);
  };
  return descriptor;
}
