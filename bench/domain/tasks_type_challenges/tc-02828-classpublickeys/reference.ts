// `keyof` on an instance type already omits private and protected members.
type ClassPublicKeys<T> = keyof T
