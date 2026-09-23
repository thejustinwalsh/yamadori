// C3 / C5 are counters modulo 3 and 5; an empty counter means "divisible".
type FBNext<C extends unknown[], M extends number> =
  [...C, unknown]['length'] extends M ? [] : [...C, unknown]

type FBWord<A extends unknown[], B extends unknown[], I> =
  A extends []
    ? B extends [] ? 'FizzBuzz' : 'Fizz'
    : B extends [] ? 'Buzz' : `${I & number}`

type FizzBuzz<
  N extends number,
  Acc extends string[] = [],
  C3 extends unknown[] = [],
  C5 extends unknown[] = [],
> = Acc['length'] extends N
  ? Acc
  : FizzBuzz<
      N,
      [...Acc, FBWord<FBNext<C3, 3>, FBNext<C5, 5>, [...Acc, unknown]['length']>],
      FBNext<C3, 3>,
      FBNext<C5, 5>
    >
