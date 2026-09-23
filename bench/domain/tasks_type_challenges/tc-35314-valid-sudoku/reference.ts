type VSDigits = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
type VSIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
type VSTriple = 0 | 1 | 2
type VSBands = [0 | 1 | 2, 3 | 4 | 5, 6 | 7 | 8]

// Nine cells are valid iff they hold every digit 1-9 (so none repeats) and nothing else.
type VSUnitOk<Cells> = VSDigits extends Cells ? ([Cells] extends [VSDigits] ? true : false) : false

type VSChecks<M extends number[][]> =
  | { [R in VSIndex]: VSUnitOk<M[R][VSIndex]> }[VSIndex]
  | { [C in VSIndex]: VSUnitOk<M[VSIndex][C]> }[VSIndex]
  | { [BR in VSTriple]: { [BC in VSTriple]: VSUnitOk<M[VSBands[BR]][VSBands[BC]]> }[VSTriple] }[VSTriple]

type ValidSudoku<M extends number[][]> = false extends VSChecks<M> ? false : true
