// Object.entries + per-entry async mapping + Object.fromEntries; type written with a helper alias.
type Resolved<T> = { -readonly [K in keyof T]: Awaited<T[K]> };

export const props = async <T extends object>(obj: T): Promise<Resolved<T>> =>
  Object.fromEntries(
    await Promise.all(Object.entries(obj).map(async ([key, value]) => [key, await value] as const)),
  ) as Resolved<T>;
