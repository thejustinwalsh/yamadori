---
name: typescript-strictness
description: Practical TypeScript rules for application code. Use when writing or reviewing TypeScript modules, types or tsconfig settings.
license: MIT
---
# TypeScript strictness

These notes collect the habits that keep a TypeScript codebase honest.

## Compiler settings

Turn on `strict` in tsconfig.json so that null and undefined are checked everywhere and implicit any is an error.

Enable `noUncheckedIndexedAccess` when a codebase indexes arrays or records by computed keys, because every indexed read then includes undefined in its type.

## Modelling data

Model a value that can take one of several shapes as a discriminated union with a literal `kind` field, and switch on that field.

Use `satisfies` to check an object literal against a type while keeping the literal's narrower inferred type.

You must never use the `any` type to silence an error, because it disables checking for every value it touches.

Avoid enums in libraries that other packages consume; prefer a union of string literals.

```ts
type Shape =
  | { kind: "circle"; r: number }
  | { kind: "square"; side: number };

function area(s: Shape): number {
  switch (s.kind) {
    case "circle": return Math.PI * s.r ** 2;
    case "square": return s.side ** 2;
  }
}
```

TypeScript types are erased at runtime, so validate untrusted input at the boundary instead of trusting a type assertion.
