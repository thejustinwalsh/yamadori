import type { Equal, Expect } from './assert_types.ts';
import type { Getters } from './solution.ts';

declare const tag: unique symbol;
type User = { name: string; age: number; URL: string; [tag]: boolean; 0: 'zero' };

type _user = Expect<Equal<Getters<User>, {
  getName: () => string;
  getAge: () => number;
  getURL: () => string;
}>>;

interface Point { x: number; y: number }
type _point = Expect<Equal<Getters<Point>, { getX: () => number; getY: () => number }>>;
type _literal = Expect<Equal<Getters<{ id: 1; kind: 'a' | 'b' }>, { getId: () => 1; getKind: () => 'a' | 'b' }>>;
type _empty = Expect<Equal<Getters<{}>, {}>>;

const g: Getters<Point> = { getX: () => 1, getY: () => 2 };
// @ts-expect-error -- getX must return number
const bad1: Getters<Point> = { getX: () => 'x', getY: () => 2 };
// @ts-expect-error -- getY is required
const bad2: Getters<Point> = { getX: () => 1 };
// @ts-expect-error -- a lower-case getter name is not part of the type
const bad3: Getters<Point> = { getX: () => 1, getY: () => 2, getx: () => 1 };
void g; void bad1; void bad2; void bad3;
