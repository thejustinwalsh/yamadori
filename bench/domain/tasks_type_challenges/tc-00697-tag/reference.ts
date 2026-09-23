declare const TAG_STORE: unique symbol

type TagIsAny<T> = 0 extends 1 & T ? true : false

type TagJoin<Tags extends string[]> =
  Tags extends [infer H extends string, ...infer R extends string[]]
    ? R extends [] ? H : `${H}\u0000${TagJoin<R>}`
    : ''

// The tag record lives under a symbol key, so it never shows up in `keyof ... & string`.
// Its value has one shared optional key ('tagged') plus one optional key per tag list;
// two different tag lists therefore stay mutually assignable, yet each carries
// [tags, untagged base] for the type-level API.
type TagRecord<Tags extends string[], Base> = {
  [TAG_STORE]?: {
    [P in 'tagged' | `tags:${TagJoin<Tags>}`]?: P extends 'tagged' ? true : [Tags, Base]
  }
}

// [tags, base] of a single (non-union) type, or never when it carries no tags.
type TagInfo<B> =
  B extends { [TAG_STORE]?: infer V }
    ? V extends { tagged?: true }
      ? Exclude<V[Exclude<keyof V, 'tagged'>], undefined> extends infer I
        ? I extends [string[], unknown] ? I : never
        : never
      : never
    : never

type TagWithBase<Base, R> = TagIsAny<Base> extends true ? R : [Base] extends [never] ? R : Base & R

type TagOne<B, T extends string> =
  [TagInfo<B>] extends [never]
    ? B & TagRecord<[T], B>
    : TagInfo<B> extends [infer Tags extends string[], infer Base]
      ? TagWithBase<Base, TagRecord<[...Tags, T], Base>>
      : never

type Tag<B, T extends string> =
  TagIsAny<B> extends true ? TagRecord<[T], B>
  : [B] extends [never] ? TagRecord<[T], B>
  : [B] extends [null | undefined] ? B
  : B extends unknown ? TagOne<B, T> : never

// Tags of each union member ([] for an untagged member).
type TagListOf<B> = B extends unknown ? ([TagInfo<B>] extends [never] ? [] : TagInfo<B>[0]) : never

// A single tag list when every member agrees, otherwise [].
type TagAgreed<U, All = U> = U extends unknown ? ([All] extends [U] ? U : []) : never

type GetTags<B> =
  TagIsAny<B> extends true ? []
  : [B] extends [never] ? []
  : TagAgreed<TagListOf<B>>

type UnTag<B> =
  B extends unknown ? ([TagInfo<B>] extends [never] ? B : TagInfo<B>[1]) : never

type TagStartsWith<Tags extends string[], T extends readonly string[]> =
  T extends readonly [infer H, ...infer R extends readonly string[]]
    ? Tags extends [infer TH, ...infer TR extends string[]]
      ? [TH] extends [H] ? ([H] extends [TH] ? TagStartsWith<TR, R> : false) : false
      : false
    : true

type TagContains<Tags extends string[], T extends readonly string[]> =
  TagStartsWith<Tags, T> extends true ? true
  : Tags extends [string, ...infer R extends string[]] ? TagContains<R, T> : false

// True when every union member satisfies the per-member check.
type TagEvery<Results> = [Results] extends [true] ? true : false

type HasTag<B, T extends string> =
  TagIsAny<B> extends true ? false
  : [B] extends [never] ? false
  : TagEvery<B extends unknown ? (T extends TagListOf<B>[number] ? true : false) : never>

type HasTags<B, T extends readonly string[]> =
  TagIsAny<B> extends true ? false
  : [B] extends [never] ? false
  : TagEvery<B extends unknown ? TagContains<TagListOf<B>, T> : never>

type HasExactTags<B, T extends readonly string[]> =
  [GetTags<B>] extends [T] ? ([T] extends [GetTags<B>] ? true : false) : false
