// A different correct shape: reduce, and a guard written with Number.isSafeInteger.
export const chunk = <T,>(items: readonly T[], size: number): T[][] => {
  if (!(Number.isSafeInteger(size) && size > 0)) throw new RangeError('size');
  return items.reduce<T[][]>((acc, x, i) => {
    if (i % size === 0) acc.push([]);
    acc[acc.length - 1]!.push(x);
    return acc;
  }, []);
};
