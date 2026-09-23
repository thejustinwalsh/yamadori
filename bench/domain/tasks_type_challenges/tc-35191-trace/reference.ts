type Trace<T extends any[][], I extends unknown[] = [], Acc = never> =
  I['length'] extends T['length']
    ? Acc
    : Trace<T, [...I, unknown], Acc | T[I['length']][I['length']]>
