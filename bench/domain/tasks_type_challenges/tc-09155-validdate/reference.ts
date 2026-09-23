type VDDigit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
type VDMonth = `0${Exclude<VDDigit, '0'>}` | '10' | '11' | '12'
type VDDay28 = `0${Exclude<VDDigit, '0'>}` | `1${VDDigit}` | `2${Exclude<VDDigit, '9'>}`
type VDDays<M extends string> =
  M extends '02' ? VDDay28
    : M extends '04' | '06' | '09' | '11' ? VDDay28 | '29' | '30'
      : VDDay28 | '29' | '30' | '31'

type ValidDate<T extends string> =
  T extends `${infer A}${infer B}${infer C}${infer D}`
    ? `${A}${B}` extends VDMonth
      ? `${C}${D}` extends VDDays<`${A}${B}`>
        ? true
        : false
      : false
    : false
