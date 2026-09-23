// The state type parameter is never used by any member, so it is structurally invisible:
// every `this`-parameter restriction is satisfied by any builder and none of the rules are enforced.
export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE';
export interface RequestSpec {
  url: string;
  method: Method;
  headers: Record<string, string>;
  body?: string;
}

interface State { url: boolean; method: Method | null }

class RequestBuilder<S extends State> {
  private constructor(
    private readonly _url: string | undefined,
    private readonly _method: Method | undefined,
    private readonly _headers: Readonly<Record<string, string>>,
    private readonly _body: string | undefined,
  ) {}

  static start(): RequestBuilder<{ url: false; method: null }> {
    return new RequestBuilder(undefined, undefined, {}, undefined);
  }

  url(this: RequestBuilder<S & { url: false }>, url: string): RequestBuilder<{ url: true; method: S['method'] }> {
    return new RequestBuilder(url, this._method, this._headers, this._body);
  }

  method<N extends Method>(this: RequestBuilder<S & { method: null }>, m: N): RequestBuilder<{ url: S['url']; method: N }> {
    return new RequestBuilder(this._url, m, this._headers, this._body);
  }

  header(name: string, value: string): RequestBuilder<S> {
    return new RequestBuilder(this._url, this._method, { ...this._headers, [name]: value }, this._body);
  }

  body(this: RequestBuilder<S & { method: 'POST' | 'PUT' | 'DELETE' }>, b: string): RequestBuilder<S> {
    return new RequestBuilder(this._url, this._method, this._headers, b);
  }

  build(this: RequestBuilder<{ url: true; method: Method }>): RequestSpec {
    return {
      url: this._url!,
      method: this._method!,
      headers: { ...this._headers },
      ...(this._body === undefined ? {} : { body: this._body }),
    };
  }
}

export const request = () => RequestBuilder.start();
