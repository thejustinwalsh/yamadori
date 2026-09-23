// Forgets to unwrap the value types: the result still claims to hold promises.
export async function props<T extends object>(obj: T): Promise<{ [K in keyof T]: T[K] }> {
  const keys = Object.keys(obj);
  const values = await Promise.all(keys.map((k) => obj[k as keyof T]));
  return Object.fromEntries(keys.map((k, i) => [k, values[i]])) as { [K in keyof T]: T[K] };
}
