// any[]-typed: compiles and runs, but the remaining parameters are lost.
export function partial<R>(fn: (...args: any[]) => R, ...head: any[]): (...tail: any[]) => R {
  return (...tail) => fn(...head, ...tail);
}
