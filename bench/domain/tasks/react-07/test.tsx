import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { useFetch } from './solution';
import { deferred, fail, flush, settle, type Deferred } from './helpers';

interface User {
  name: string;
}

interface Call {
  url: string;
  signal: AbortSignal | undefined;
  d: Deferred<Response>;
}

let calls: Call[] = [];

/** A fetch whose responses the test releases by hand. Like the real one, it
 *  rejects with an AbortError as soon as its signal is aborted. */
function stubFetch() {
  calls = [];
  const fn = vi.fn((input: string, init?: RequestInit) => {
    const d = deferred<Response>();
    const signal = init?.signal ?? undefined;
    signal?.addEventListener('abort', () =>
      d.reject(new DOMException('The operation was aborted.', 'AbortError')),
    );
    calls.push({ url: String(input), signal, d });
    return d.promise;
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

function response(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

async function respond(call: Call, status: number, body: unknown) {
  await settle(call.d, response(status, body));
  await flush();
}

function Probe({ url }: { url: string | null }) {
  const r: { data: User | undefined; error: Error | undefined; loading: boolean } =
    useFetch<User>(url);
  return (
    <div>
      <p data-testid="loading">{r.loading ? 'loading' : 'done'}</p>
      <p data-testid="data">{r.data ? r.data.name : '-'}</p>
      <p data-testid="error">{r.error ? r.error.message : '-'}</p>
    </div>
  );
}

const text = (id: string) => screen.getByTestId(id).textContent;
const state = () => [text('loading'), text('data'), text('error')];

describe('useFetch', () => {
  beforeEach(() => {
    calls = [];
  });

  it('is idle and makes no request when url is null', async () => {
    const fn = stubFetch();
    render(<Probe url={null} />);
    await flush();
    expect(fn).not.toHaveBeenCalled();
    expect(state()).toEqual(['done', '-', '-']);
  });

  it('fetches with an AbortSignal and shows the JSON body', async () => {
    stubFetch();
    render(<Probe url="/users/1" />);
    expect(calls.length).toBe(1);
    expect(calls[0]!.url).toBe('/users/1');
    expect(calls[0]!.signal).toBeInstanceOf(AbortSignal);
    expect(state()).toEqual(['loading', '-', '-']);
    await respond(calls[0]!, 200, { name: 'ada' });
    expect(state()).toEqual(['done', 'ada', '-']);
  });

  it('turns a non-ok response into an HTTP error', async () => {
    stubFetch();
    render(<Probe url="/missing" />);
    await respond(calls[0]!, 404, { message: 'not here' });
    expect(state()).toEqual(['done', '-', 'HTTP 404']);
  });

  it('reports a network failure as the error', async () => {
    stubFetch();
    render(<Probe url="/offline" />);
    await fail(calls[0]!.d, new TypeError('Failed to fetch'));
    await flush();
    expect(state()).toEqual(['done', '-', 'Failed to fetch']);
  });

  it('resets and aborts the previous request when url changes', async () => {
    stubFetch();
    const { rerender } = render(<Probe url="/users/1" />);
    await respond(calls[0]!, 200, { name: 'ada' });
    rerender(<Probe url="/users/2" />);
    await flush();
    expect(state()).toEqual(['loading', '-', '-']);
    rerender(<Probe url="/users/3" />);
    await flush();
    expect(calls[1]!.signal?.aborted).toBe(true);
    expect(state()).toEqual(['loading', '-', '-']);
  });

  it('never lets a slow old request overwrite a newer one', async () => {
    stubFetch();
    const { rerender } = render(<Probe url="/slow" />);
    rerender(<Probe url="/fast" />);
    const [slow, fast] = [calls[0]!, calls[calls.length - 1]!];
    expect(fast.url).toBe('/fast');
    await respond(fast, 200, { name: 'bob' });
    expect(state()).toEqual(['done', 'bob', '-']);
    // If the answer aborted it, this resolve is a no-op on a rejected promise.
    await act(async () => {
      slow.d.resolve(response(200, { name: 'stale' }));
    });
    await flush();
    expect(state()).toEqual(['done', 'bob', '-']);
  });

  it('goes idle and aborts when url becomes null', async () => {
    stubFetch();
    const { rerender } = render(<Probe url="/users/1" />);
    rerender(<Probe url={null} />);
    await flush();
    expect(calls[0]!.signal?.aborted).toBe(true);
    expect(state()).toEqual(['done', '-', '-']);
  });

  it('aborts the in-flight request on unmount', async () => {
    stubFetch();
    const { unmount } = render(<Probe url="/users/1" />);
    unmount();
    await flush();
    expect(calls[0]!.signal?.aborted).toBe(true);
  });
});
