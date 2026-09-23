type JSONSchemaFlatten<O> = { [K in keyof O]: O[K] }

type JSONSchemaObject<P, Req> = JSONSchemaFlatten<
  { [K in keyof P as K extends Req ? K : never]: JSONSchema2TS<P[K]> }
  & { [K in keyof P as K extends Req ? never : K]?: JSONSchema2TS<P[K]> }
>

type JSONSchema2TS<T> =
  T extends { enum: infer E extends readonly unknown[] } ? E[number]
  : T extends { type: 'string' } ? string
  : T extends { type: 'number' } ? number
  : T extends { type: 'boolean' } ? boolean
  : T extends { type: 'array' }
    ? T extends { items: infer I } ? JSONSchema2TS<I>[] : unknown[]
  : T extends { type: 'object' }
    ? T extends { properties: infer P }
      ? JSONSchemaObject<P, T extends { required: infer R extends readonly unknown[] } ? R[number] : never>
      : Record<string, unknown>
  : never
