type MyUppercaseTable = {
  a: 'A', b: 'B', c: 'C', d: 'D', e: 'E', f: 'F', g: 'G', h: 'H', i: 'I',
  j: 'J', k: 'K', l: 'L', m: 'M', n: 'N', o: 'O', p: 'P', q: 'Q', r: 'R',
  s: 'S', t: 'T', u: 'U', v: 'V', w: 'W', x: 'X', y: 'Y', z: 'Z',
}

type MyUppercase<T extends string, Acc extends string = ''> =
  T extends `${infer F}${infer R}`
    ? MyUppercase<R, `${Acc}${F extends keyof MyUppercaseTable ? MyUppercaseTable[F] : F}`>
    : Acc
