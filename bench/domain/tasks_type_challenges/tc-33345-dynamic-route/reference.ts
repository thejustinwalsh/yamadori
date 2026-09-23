type DRSegments<S extends string> =
  S extends `${infer H}/${infer R}` ? [H, ...DRSegments<R>] : [S]

// What one path segment contributes.
type DRKind<Seg extends string> =
  Seg extends '' ? ['skip']
  : Seg extends `[[...${infer N}]]` ? (N extends '' | `${string}${'[' | ']' | '/'}${string}` ? ['static'] : ['optionalCatchAll', N])
  : Seg extends `[...${infer N}]` ? (N extends '' ? ['dynamic', '...'] : N extends `${string}${'[' | ']'}${string}` ? ['static'] : ['catchAll', N])
  : Seg extends `[${infer N}]` ? (N extends '' | `${string}${'[' | ']'}${string}` ? ['static'] : ['dynamic', N])
  : ['static']

type DRMerge<O> = { [K in keyof O]: O[K] }

// Open: a catch-all was seen with no static segment after it yet.
type DRGo<Segs extends string[], Req = {}, Opt = {}, Open extends boolean = false> =
  Segs extends [infer S extends string, ...infer R extends string[]]
    ? DRKind<S> extends ['dynamic', infer N extends string]
      ? DRGo<R, Req & { [K in N]: string }, Opt, Open>
      : DRKind<S> extends ['catchAll', infer N extends string]
        ? Open extends true ? never : DRGo<R, Req & { [K in N]: string[] }, Opt, true>
        : DRKind<S> extends ['optionalCatchAll', infer N extends string]
          ? Open extends true ? never : DRGo<R, Req, Opt & { [K in N]?: string[] }, true>
          : DRKind<S> extends ['static']
            ? DRGo<R, Req, Opt, false>
            : DRGo<R, Req, Opt, Open>
    : DRMerge<Req & Opt>

type DynamicRoute<T extends string> = DRGo<DRSegments<T>>
