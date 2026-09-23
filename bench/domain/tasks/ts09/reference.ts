export function partial<H extends unknown[], T extends unknown[], R>(
  fn: (...args: [...H, ...T]) => R,
  ...head: H
): (...tail: T) => R {
  return (...tail: T) => fn(...head, ...tail);
}
