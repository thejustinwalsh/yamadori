// `1 & T` is `never` for every T except `any`, where it stays `any` and so accepts 0.
type IsAny<T> = 0 extends 1 & T ? true : false
