// body() is always available, so a GET request with a body (or a body before any method) compiles.
export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE';

export type RequestSpec = {
  url: string;
  method: Method;
  headers: Record<string, string>;
  body?: string;
};

type Draft = { url?: string; method?: Method; headers: Record<string, string>; body?: string };

export type Builder<HasUrl extends boolean, M extends Method | undefined> =
  { header(name: string, value: string): Builder<HasUrl, M> }
  & (HasUrl extends false ? { url(url: string): Builder<true, M> } : {})
  & (M extends undefined ? { method<N extends Method>(method: N): Builder<HasUrl, N> } : {})
  & { body(body: string): Builder<HasUrl, M> }
  & (HasUrl extends true ? (M extends Method ? { build(): RequestSpec } : {}) : {});

function make(d: Draft): Builder<boolean, Method | undefined> {
  const next = (patch: Partial<Draft>) => make({ ...d, ...patch, headers: { ...d.headers, ...patch.headers } });
  const b = {
    url: (url: string) => next({ url }),
    method: (method: Method) => next({ method }),
    header: (name: string, value: string) => next({ headers: { [name]: value } }),
    body: (body: string) => next({ body }),
    build: (): RequestSpec => {
      const spec: RequestSpec = { url: d.url!, method: d.method!, headers: { ...d.headers } };
      if (d.body !== undefined) spec.body = d.body;
      return spec;
    },
  };
  return b as unknown as Builder<boolean, Method | undefined>;
}

export function request(): Builder<false, undefined> {
  return make({ headers: {} }) as unknown as Builder<false, undefined>;
}
