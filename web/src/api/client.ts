// The one way the dashboard reads the server. Mirrors the Python pages
// (mcp/dash_vitals.py, mcp/dashboard.py): the key lives in this browser only,
// under localStorage['yamadori_key'], sent as `Authorization: Bearer <key>`.
// It is the same key an editor sends; there is no dashboard account.
//
// Every failure is typed and carries what actually happened -- the HTTP
// status, the server's own error text, the exception -- because each panel
// prints it verbatim. "Something went wrong" is not a state.

export const KEY_STORAGE = 'yamadori_key';

export function readKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? '';
  } catch {
    return '';
  }
}

export function writeKey(key: string): void {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage blocked: the sign-in form says so via the next 401 */
  }
  window.dispatchEvent(new Event('yamadori:key'));
}

export type Failure =
  | { kind: 'nokey' }
  | { kind: 'unauthorised'; message: string }
  | { kind: 'http'; status: number; message: string }
  | { kind: 'network'; message: string }
  | { kind: 'payload'; message: string };

export type Fetched<T> = { ok: true; data: T; receivedAt: number } | { ok: false; failure: Failure };

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<Fetched<T>> {
  const key = readKey();
  if (!key) return { ok: false, failure: { kind: 'nokey' } };
  let res: Response;
  try {
    res = await fetch(path, { headers: { Authorization: `Bearer ${key}` }, cache: 'no-store', signal });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    return { ok: false, failure: { kind: 'network', message: `cannot reach ${path}: ${String(e)}` } };
  }
  let body: unknown = null;
  const text = await res.text();
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  const serverMessage = (() => {
    const e = (body as { error?: unknown } | null)?.error;
    if (typeof e === 'string') return e;
    if (e && typeof e === 'object' && 'message' in e) return String((e as { message: unknown }).message);
    return '';
  })();
  if (res.status === 401) {
    return { ok: false, failure: { kind: 'unauthorised', message: serverMessage || 'the key was not accepted' } };
  }
  if (!res.ok) {
    return { ok: false, failure: { kind: 'http', status: res.status, message: serverMessage || text.slice(0, 300) } };
  }
  if (body === null) {
    return { ok: false, failure: { kind: 'payload', message: `${path} answered ${res.status} with no JSON body` } };
  }
  if (typeof (body as { error?: unknown }).error === 'string') {
    return { ok: false, failure: { kind: 'payload', message: serverMessage } };
  }
  return { ok: true, data: body as T, receivedAt: Date.now() };
}

/**
 * One line: what failed and the server's own words for it, trimmed. The
 * dashboard states what is missing; it does not explain or advise.
 */
export function describeFailure(path: string, f: Failure): { title: string; detail: string } {
  const trim = (m: string) => (m.length > 160 ? `${m.slice(0, 157)}…` : m);
  switch (f.kind) {
    case 'nokey':
      return { title: 'not signed in', detail: '' };
    case 'unauthorised':
      return { title: `${path} · HTTP 401`, detail: trim(f.message) };
    case 'http':
      return { title: `${path} · HTTP ${f.status}`, detail: trim(f.message) };
    case 'network':
      return { title: `${path} · no response`, detail: '' };
    case 'payload':
      return { title: `${path} · error`, detail: trim(f.message) };
  }
}

export type Refusal = { what?: string; field?: string; why?: string };

export type Posted<T> =
  | { ok: true; data: T }
  | { ok: false; failure: Failure; reasons: Refusal[] };

/**
 * POST a JSON body to a /dash/api route. The dataset routes answer
 * {ok: false, error, reasons} with 400/409 for a refusal; the error text and
 * every reason come back so the form can print them verbatim.
 */
export async function postJson<T>(path: string, body: unknown): Promise<Posted<T>> {
  return sendJson<T>('POST', path, body);
}

/** PUT a JSON body to a /dash/api route; the same answer shape as postJson. */
export async function putJson<T>(path: string, body: unknown): Promise<Posted<T>> {
  return sendJson<T>('PUT', path, body);
}

async function sendJson<T>(method: 'POST' | 'PUT', path: string, body: unknown): Promise<Posted<T>> {
  const key = readKey();
  if (!key) return { ok: false, failure: { kind: 'nokey' }, reasons: [] };
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });
  } catch (e) {
    return { ok: false, failure: { kind: 'network', message: `cannot reach ${path}: ${String(e)}` }, reasons: [] };
  }
  const text = await res.text();
  let parsed: { ok?: unknown; error?: unknown; reasons?: unknown } | null = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = null;
  }
  const message = typeof parsed?.error === 'string' ? parsed.error : text.slice(0, 300);
  const reasons = Array.isArray(parsed?.reasons) ? (parsed.reasons as Refusal[]) : [];
  if (res.status === 401) return { ok: false, failure: { kind: 'unauthorised', message: message || 'the key was not accepted' }, reasons };
  if (!res.ok || !parsed || parsed.ok !== true) {
    if (!parsed) return { ok: false, failure: { kind: 'payload', message: `${path} answered ${res.status} with no JSON body` }, reasons };
    return { ok: false, failure: { kind: 'http', status: res.status, message }, reasons };
  }
  return { ok: true, data: parsed as T };
}
