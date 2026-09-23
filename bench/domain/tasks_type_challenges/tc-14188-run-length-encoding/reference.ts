namespace RLE {
  type Digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'

  type Flush<P extends string, C extends unknown[]> =
    C['length'] extends 0 ? '' : C['length'] extends 1 ? P : `${C['length']}${P}`

  type Repeat<C extends string, N extends number, Cnt extends unknown[] = [], Acc extends string = ''> =
    Cnt['length'] extends N ? Acc : Repeat<C, N, [...Cnt, unknown], `${Acc}${C}`>

  type ToCount<D extends string> =
    D extends '' ? 1 : D extends `${infer N extends number}` ? N : 1

  export type Encode<S extends string, Prev extends string = '', Cnt extends unknown[] = [], Out extends string = ''> =
    S extends `${infer F}${infer R}`
      ? F extends Prev
        ? Encode<R, Prev, [...Cnt, unknown], Out>
        : Encode<R, F, [unknown], `${Out}${Flush<Prev, Cnt>}`>
      : `${Out}${Flush<Prev, Cnt>}`

  export type Decode<S extends string, D extends string = '', Out extends string = ''> =
    S extends `${infer F}${infer R}`
      ? F extends Digit
        ? Decode<R, `${D}${F}`, Out>
        : Decode<R, '', `${Out}${Repeat<F, ToCount<D>>}`>
      : Out
}
