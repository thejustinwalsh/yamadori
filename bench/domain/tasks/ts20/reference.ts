export async function props<T extends object>(obj: T): Promise<{ [K in keyof T]: Awaited<T[K]> }> {
  const keys = Object.keys(obj) as (keyof T)[];
  const values = await Promise.all(keys.map((k) => obj[k]));
  const out = {} as { [K in keyof T]: Awaited<T[K]> };
  keys.forEach((k, i) => {
    out[k] = values[i] as Awaited<T[typeof k]>;
  });
  return out;
}
