// Shared test doubles for the react task graders (copied next to each task's
// test.tsx as ./helpers). They stand in for browser APIs jsdom does not
// implement, and they are controllable, so a test can drive the browser side.
import { vi } from 'vitest';
import { act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

export function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Resolve/reject inside act() and let React flush what follows. */
export async function settle<T>(d: Deferred<T>, value: T): Promise<void> {
  await act(async () => {
    d.resolve(value);
    await d.promise;
  });
}

export async function fail<T>(d: Deferred<T>, reason: unknown): Promise<void> {
  await act(async () => {
    d.reject(reason);
    await d.promise.catch(() => undefined);
  });
}

/** Let pending microtasks and React work flush. */
export async function flush(): Promise<void> {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

/** Fake timers + a userEvent instance that advances them. */
export function fakeTimerUser() {
  vi.useFakeTimers();
  return userEvent.setup({ advanceTimers: (ms) => vi.advanceTimersByTime(ms) });
}

export function advance(ms: number): void {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

// ------------------------------------------------ IntersectionObserver ---

export class MockIntersectionObserver implements IntersectionObserver {
  static instances: MockIntersectionObserver[] = [];
  readonly root: Element | Document | null;
  readonly rootMargin: string;
  readonly thresholds: ReadonlyArray<number>;
  readonly scrollMargin: string = '0px';
  readonly callback: IntersectionObserverCallback;
  readonly options: IntersectionObserverInit | undefined;
  readonly elements = new Set<Element>();
  disconnected = false;

  constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
    this.callback = callback;
    this.options = options;
    this.root = options?.root ?? null;
    this.rootMargin = options?.rootMargin ?? '0px';
    const t = options?.threshold ?? 0;
    this.thresholds = Array.isArray(t) ? t : [t as number];
    MockIntersectionObserver.instances.push(this);
  }
  observe(el: Element): void {
    this.elements.add(el);
  }
  unobserve(el: Element): void {
    this.elements.delete(el);
  }
  disconnect(): void {
    this.elements.clear();
    this.disconnected = true;
  }
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
}

export interface IOControl {
  /** Elements currently observed by any live observer. */
  observed(): Element[];
  /** Fire an entry for `el` on every observer watching it (inside act). */
  trigger(el: Element, isIntersecting: boolean): void;
  instances(): MockIntersectionObserver[];
}

export function installIntersectionObserver(): IOControl {
  MockIntersectionObserver.instances = [];
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
  return {
    observed() {
      const out: Element[] = [];
      for (const o of MockIntersectionObserver.instances) out.push(...o.elements);
      return out;
    },
    instances() {
      return MockIntersectionObserver.instances;
    },
    trigger(el, isIntersecting) {
      act(() => {
        for (const o of MockIntersectionObserver.instances) {
          if (!o.elements.has(el)) continue;
          const rect = el.getBoundingClientRect();
          const entry = {
            target: el,
            isIntersecting,
            intersectionRatio: isIntersecting ? 1 : 0,
            boundingClientRect: rect,
            intersectionRect: rect,
            rootBounds: null,
            time: Date.now(),
          } as IntersectionObserverEntry;
          o.callback([entry], o);
        }
      });
    },
  };
}

// ---------------------------------------------------------- matchMedia ---

type MqlListener = (ev: MediaQueryListEvent) => void;

export interface MediaControl {
  /** Change whether `query` matches and notify its listeners (inside act). */
  set(query: string, matches: boolean): void;
  /** Live change listeners currently registered for `query`. */
  listeners(query: string): number;
}

export function installMatchMedia(initial: Record<string, boolean> = {}): MediaControl {
  const state = new Map<string, boolean>(Object.entries(initial));
  const listeners = new Map<string, Set<MqlListener>>();
  const bucket = (q: string) => {
    let s = listeners.get(q);
    if (!s) listeners.set(q, (s = new Set()));
    return s;
  };
  const matchMedia = (query: string): MediaQueryList => {
    const mql = {
      media: query,
      get matches() {
        return state.get(query) ?? false;
      },
      onchange: null,
      addEventListener(type: string, fn: MqlListener) {
        if (type === 'change') bucket(query).add(fn);
      },
      removeEventListener(type: string, fn: MqlListener) {
        if (type === 'change') bucket(query).delete(fn);
      },
      addListener(fn: MqlListener) {
        bucket(query).add(fn);
      },
      removeListener(fn: MqlListener) {
        bucket(query).delete(fn);
      },
      dispatchEvent() {
        return true;
      },
    };
    return mql as unknown as MediaQueryList;
  };
  vi.stubGlobal('matchMedia', matchMedia);
  return {
    set(query, matches) {
      state.set(query, matches);
      act(() => {
        for (const fn of [...bucket(query)]) {
          fn({ matches, media: query } as MediaQueryListEvent);
        }
      });
    },
    listeners(query) {
      return bucket(query).size;
    },
  };
}
