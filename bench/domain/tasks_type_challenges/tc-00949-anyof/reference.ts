type AnyOfFalsy = 0 | '' | false | [] | { [key: string]: never } | undefined | null

type AnyOf<T extends readonly unknown[]> = T[number] extends AnyOfFalsy ? false : true
