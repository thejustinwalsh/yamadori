// Puts the pre-supplied arguments AFTER the ones given later.
export function partial<H extends unknown[], T extends unknown[], R>(
  fn: (...args: [...H, ...T]) => R,
  ...head: H
): (...tail: T) => R {
  return (...tail: T) => (fn as (...a: unknown[]) => R)(...tail, ...head);
}
