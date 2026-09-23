type Digits = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9

// A grid is 9 rows; each row is 3 groups of 3 cells.
type SDRowIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
type SDTriple = 0 | 1 | 2
type SDBands = [0 | 1 | 2, 3 | 4 | 5, 6 | 7 | 8]

// A unit of nine cells is solved iff it holds every digit.
type SDUnitOk<Cells> = Digits extends Cells ? true : false

type SDChecks<G extends Digits[][][]> =
  | { [R in SDRowIndex]: SDUnitOk<G[R][number][number]> }[SDRowIndex]
  | { [Gi in SDTriple]: { [I in SDTriple]: SDUnitOk<G[number][Gi][I]> }[SDTriple] }[SDTriple]
  | { [B in SDTriple]: { [Gi in SDTriple]: SDUnitOk<G[SDBands[B]][Gi][number]> }[SDTriple] }[SDTriple]

type SudokuSolved<G extends Digits[][][]> = false extends SDChecks<G> ? false : true
