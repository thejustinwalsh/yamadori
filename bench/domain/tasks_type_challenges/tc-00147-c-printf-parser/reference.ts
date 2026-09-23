type ControlsMap = {
  c: 'char'
  s: 'string'
  d: 'dec'
  o: 'oct'
  h: 'hex'
  f: 'float'
  p: 'pointer'
}

type ParsePrintFormat<S extends string = '', Acc extends string[] = []> =
  S extends `${string}%${infer C}${infer Rest}`
    ? C extends keyof ControlsMap
      ? ParsePrintFormat<Rest, [...Acc, ControlsMap[C]]>
      : ParsePrintFormat<Rest, Acc>
    : Acc
