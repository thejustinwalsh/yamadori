# The tools API (`:1235`) and its key

`mcp/tools_api.py` serves the code-intelligence tools over HTTP: REST routes
for your own scripts and `POST /mcp` for MCP clients, the same
`code_search.handle()` the stdio server runs. It binds `0.0.0.0` on purpose,
so remote MCP clients on the operator's laptop and phone can reach it over
ZeroTier. MCP is a bonus surface. Nothing in the stack needs it: the proxy
and the second brain call `code_search` in-process and never use `:1235`.

Until 2026-09-26 it had **no auth**. Anything on the LAN (10.0.0.44) or on
ZeroTier could call `read_file_range` and `run_check`. So could any web page
open in the operator's browser, through DNS rebinding. The probe in
`docs/SELF-IMPROVEMENT-LOG.md` #48 found it answering 200 with index stats
and no key. The operator decided (2026-09-26) to require a key.

## The door

Every request passes two checks, in this order. Each follows the MCP spec,
revision 2026-07-28.

**1. Origin, on every route** (`/health` and the metadata included).
[Streamable HTTP, "Security & Endpoint"](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#security-%26-endpoint):
*"Servers MUST validate the `Origin` header on all incoming connections"*,
and a present, invalid Origin gets 403.

| Origin header | result |
|---|---|
| absent (curl, SDKs, every harness) | passes this check |
| in `YAMADORI_TOOLS_ALLOWED_ORIGINS` | passes, and gets a CORS grant for that origin (never `*`) |
| anything else, `null` included | **403**. On `/mcp` the body is a JSON-RPC error with no `id`. |

`YAMADORI_TOOLS_ALLOWED_ORIGINS` is a comma-separated list, read on each
request. Matching ignores case, a trailing slash and a default port. **It is
empty by default.** The dashboard is served by `:1234` and fetches only
`:1234`'s own relative paths (`web/src/api/client.ts`), so no browser page in
this repo calls `:1235`. `YAMADORI_PUBLIC_BASE` is therefore not on the list.

**2. The key, on everything except `/health` and
`/.well-known/oauth-protected-resource[/...]`.**
[Authorization, "Access Token Usage"](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization#access-token-usage):
the key goes in `Authorization: Bearer <key>`, in every HTTP request, and
never in the query string.

- **The key is the `:1234` account key.** It is checked by
  `accounts.identify()`; mint one with
  `python mcp/accounts.py create <label>`. Nothing new is stored.
- **A missing or wrong key gets 401** with
  `WWW-Authenticate: Bearer resource_metadata="<metadata URL>"`
  ([RFC 9728 §5.1](https://www.rfc-editor.org/rfc/rfc9728.html#section-5.1)).
  `error="invalid_token"` is added only when a key was presented
  ([RFC 6750 §3.1](https://datatracker.ietf.org/doc/html/rfc6750#section-3.1)).
  On `/mcp` the body is a JSON-RPC error with no `id`. On a REST route it has
  the tools' usual shape: `error`, `message`, `retryable: false`, `remedy`.
- **With no `accounts.json`, the tools API is closed, not open.** The proxy
  treats a missing registry as single-user and open. A door that reads files
  should not be open just because nobody made a key. An unreadable registry
  is closed as well, exactly as on `:1234`.
- **No other place for the key is accepted.** A key in the query string
  (`?access_token=`, `?key=`) is not read. Neither is `X-API-Key`: no in-repo
  client needs it, and the spec names Bearer.

Other changes in the same step:

- `GET /health` is liveness only: `{"ok": true}`. It used to include the
  index statistics. `scripts/watchdog.ps1` and `scripts/deploy_check.py` read
  only the status code (PROTOCOL rule 12). The statistics are `GET /status`,
  which needs the key.
- The access log drops query strings, which carry search text and any key a
  client put there by mistake, and redacts anything key-shaped. A refusal is
  logged with its reason (`refused 401 POST /mcp: unrecognised API key`),
  never with the key.
- A POST that is refused is never parsed: its body is drained up to 1 MiB and
  the connection closes. A body over 32 MiB is refused with 413 and not read.
- `mcp/mcp_ws.py` (the websocket MCP on `:1236`) now has the same gate before
  the upgrade. Nothing starts it today, but it also bound `0.0.0.0` with no
  auth.

### Protected Resource Metadata (RFC 9728), and the gap

These are served without a key:

| URL | `resource` |
|---|---|
| `<base>/.well-known/oauth-protected-resource` | `<base>` (the REST routes' challenge points here) |
| `<base>/.well-known/oauth-protected-resource/mcp` | `<base>/mcp` (`/mcp`'s challenge points here) |

Each document's `resource` is the URL it describes
([RFC 9728 §3.3](https://www.rfc-editor.org/rfc/rfc9728.html#section-3.3)).
That is why `/mcp` gets the path-inserted document. Both documents carry
`bearer_methods_supported: ["header"]` and `resource_name`.

`<base>` is `YAMADORI_TOOLS_PUBLIC_BASE` when set. Otherwise it is
`http://<Host>`, using the request's Host header after validation. A Host that
is not a plain host falls back to loopback, so it cannot write into the
challenge. With a base that has a path, such as
`https://ai.thejustinwalsh.me/tools`, the well-known suffix goes between the
host and the path (RFC 9728 §3.1).

**The gap: there is no authorization server, so the document names none.**
`authorization_servers` is omitted. RFC 9728 §2 makes it OPTIONAL. The
[MCP spec's Authorization Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/authorization-server-discovery)
says the document *"MUST include the `authorization_servers` field containing
at least one authorization server"*. So **this server does not conform to MCP
authorization**. The key is an operator-issued API key, not an OAuth access
token. Its audience is not bound as RFC 8707 requires, and nothing issues or
refreshes it. The spec allows this state: authorization is *"OPTIONAL for MCP
implementations"* (Authorization, "Protocol Requirements"), and the transport
only says servers *"SHOULD implement proper authentication for all
connections"*.

What the gap means in practice:

- A client configured with the header never reaches the OAuth path. The
  request succeeds, so no 401 starts discovery. Claude Code documents that
  with `headers.Authorization` set, a 401 is reported as a failed connection,
  with no fallback to OAuth.
- A client with no key that tries OAuth discovery gets the metadata, finds no
  authorization server, and stops. It fails at discovery, not partway through
  a login flow that cannot finish.
- **Closing the gap** means running an OAuth 2.1 authorization server (RFC
  8414 metadata, PKCE, RFC 8707 audience binding) and listing it in the
  document. That is an operator decision and has not been built.

**Behind Caddy**, `https://ai.thejustinwalsh.me/tools/*` reaches `:1235` with
the prefix stripped. The metadata's well-known URL for that base is
`https://ai.thejustinwalsh.me/.well-known/oauth-protected-resource/tools/mcp`.
That path falls under Caddy's catch-all, which goes to the proxy on `:1234`,
so it is not served there. Only discovery depends on that URL, and discovery
has nothing to discover yet (see the gap above). Keyed requests through
`/tools/` are unaffected: Caddy passes the `Authorization` header through.

### The MCP protocol version `/mcp` speaks

`/mcp` speaks the **legacy, initialize-era** protocol:
`code_search._handle` answers `initialize` with
`"protocolVersion": "2024-11-05"`. It is served as a single POST endpoint that
returns one JSON body per request and 202 for a notification. It has no SSE,
no `Mcp-Session-Id`, no `MCP-Protocol-Version` or `Mcp-Method` checks, and no
405 on GET. It is not the 2026-07-28 stateless shape. This step did not
change the protocol. The door runs **before** `handle()`, so an
unauthenticated request is refused whichever era the client speaks, and
`initialize` itself needs the key.

## Configuring a client

Every snippet reads the key from an environment variable,
`YAMADORI_API_KEY`: the variable this repo's client docs already use
(`docs/HARNESS-RESPONSES.md`). Never paste a key into a checked-in file.

**Claude Code** (`.mcp.json`, or `claude mcp add`). The format and `${VAR}`
expansion in `url` and `headers` are from
[code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp):

```json
{
  "mcpServers": {
    "yamadori-tools": {
      "type": "http",
      "url": "http://ai.thejustinwalsh.me:1235/mcp",
      "headers": { "Authorization": "Bearer ${YAMADORI_API_KEY}" }
    }
  }
}
```

```bash
claude mcp add --transport http yamadori-tools http://ai.thejustinwalsh.me:1235/mcp \
  --header "Authorization: Bearer $YAMADORI_API_KEY"
```

(The CLI form expands the variable in your shell, so the stored config holds
the value. Prefer the `.mcp.json` form, which keeps the reference.)

**OpenCode** (`opencode.json`, key `mcp`). The shape is `McpRemoteConfig` in
`@opencode-ai/sdk` 1.18.32 (`dist/v2/gen/types.gen.d.ts`: `type: "remote"`,
`url`, `headers`, `oauth`). `{env:VAR}` substitution is from
[opencode.ai/docs/mcp-servers](https://opencode.ai/docs/mcp-servers/).
`"oauth": false` stops OpenCode from starting OAuth on a 401 (there is no
authorization server to find):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "yamadori-tools": {
      "type": "remote",
      "url": "http://ai.thejustinwalsh.me:1235/mcp",
      "headers": { "Authorization": "Bearer {env:YAMADORI_API_KEY}" },
      "oauth": false
    }
  }
}
```

**Hermes** (`~/.hermes/config.yaml`, key `mcp_servers`). `url` and `headers`
are read in `tools/mcp_tool_server_run.py`, and `${VAR}` / `${env:VAR}` is
interpolated by `tools/mcp_tool_config.py` `_load_mcp_config` (hermes-agent
`ee5ee84`). `.env` is loaded first, so the variable can live in
`~/.hermes/.env`:

```yaml
mcp_servers:
  yamadori-tools:
    url: "http://ai.thejustinwalsh.me:1235/mcp"
    headers:
      Authorization: "Bearer ${env:YAMADORI_API_KEY}"
```

**A stdio-only client** runs `mcp/mcp_bridge.py` locally. It reads
`YAMADORI_API_KEY` (or `--key-file PATH`) and sends it as the Bearer header:

```json
{ "mcpServers": { "yamadori-tools": {
    "command": "python",
    "args": ["mcp_bridge.py", "--url", "http://ai.thejustinwalsh.me:1235/mcp"],
    "env": { "YAMADORI_API_KEY": "${YAMADORI_API_KEY}" } } } }
```

**curl / your own code:**

```bash
curl http://ai.thejustinwalsh.me:1235/mcp \
  -H "Authorization: Bearer $YAMADORI_API_KEY" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
curl "http://ai.thejustinwalsh.me:1235/definition?symbol=parse_tool_call" \
  -H "Authorization: Bearer $YAMADORI_API_KEY"
```

## Tests

- **Offline:** `mcp/test_tools_api_auth.py` runs against a real
  `ThreadingHTTPServer` on an ephemeral loopback port, with a temp registry
  and a temp index. On the unchanged code it scored 23/109: every request
  without a key got 200. After the change it scores 118/118. It checks:
  - 401 and the challenge on every route
  - `invalid_token` only when a key was presented
  - no key accepted from the query string or `X-API-Key`
  - a good key gets 200 on `/mcp` `tools/list` and on the REST routes
  - both metadata documents, including under a prefixed base
  - 403 for a foreign or `null` Origin on every route, including preflight,
    even with a valid key
  - an allowed Origin with a CORS grant, and no `*`
  - `/health` is liveness only
  - no registry, or an unreadable one, is closed
  - body limits
  - the key never reaches the log
  - the bridge sends the key
  - the websocket gate
- **Live, written but not run:** `mcp/test_tools_live.py`
  `test_tools_api_door_live` sends the key from `YAMADORI_TEST_KEY` or
  `--key-file`, as `scripts/run_tests.py --live` passes it, to the *running*
  `:1235`. It checks: no key gives 401 with the challenge, a foreign Origin
  gives 403, the key gives 200, `/health` is liveness only, and the metadata
  is served. Until the tools API restarts on this code, its first check fails
  with HTTP 200, and that failure is the hole being reported. The operator's
  rule is fix, test, deploy together (`scripts/deploy_check.py` after the
  restart).
