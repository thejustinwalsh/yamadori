// Two cockpit panels over what serves the model and what runs beside it:
//
//   SERVING · MAX MODE      /dash/api/vitals `serving` (mcp/vitals.py over
//                           mcp/max_mode.py and llama-swap's /running): which
//                           main model holds the 5060 Ti, what is loaded, and
//                           which tier the max model serves
//   MCP · HOSTED SERVERS    /dash/api/mcp (mcp/mcp_host.py status()): the MCP
//                           servers the proxy runs for the model, read-only
//
// Each states what is missing in the server's own words; none advises.
import * as stylex from '@stylexjs/stylex';
import { MCP_PATH, mcpState, type Mcp, type McpServer } from '../api/host';
import type { Failure } from '../api/client';
import type { LoadedModel, Tiers, Vitals } from '../api/types';
import { usePoll, useNow } from '../api/usePoll';
import { ago, n } from '../format';
import { colors } from '../tokens/tokens.stylex';
import { Panel } from './Panel';
import { Chip, Label, layout, Stat, type Tone } from './primitives';
import { PollState, StateView } from './StateView';
import { Table } from './Table';
import { text } from './text';

const s = stylex.create({
  note: { margin: 0, color: colors.onSurfaceVariant },
  tool: { margin: 0, color: colors.onSurfaceVariant, textTransform: 'none' },
  name: { color: colors.primary },
});

type P = { fill?: boolean };

const STATE_TONE: Record<string, Tone> = {
  ready: 'moss',
  starting: 'cyan',
  stopped: 'muted',
  'not started': 'muted',
  disabled: 'muted',
  exited: 'crimson',
  failed: 'crimson',
};

export function ServingPanel({
  v,
  tiers,
  stale,
  failure,
  fill,
}: P & { v: Vitals | null; tiers: Tiers | null; stale?: boolean; failure: Failure | null }) {
  const now = useNow(1000) / 1000;
  const sv = v?.serving;
  const maxTier = tiers?.max_mode?.tier ?? sv?.max_tier ?? 'max';
  const tag = !sv ? '—' : sv.enabled ? (sv.max_active ? 'MAX MODE HOLDS THE CARD' : 'MAX MODE CONFIGURED') : 'ONE MODEL';
  return (
    <Panel kanji="主" title="SERVING · MAX MODE" tag={tag} tagTone={sv?.max_active ? 'rose' : 'moss'} stale={stale} fill={fill}>
      {!v ? (
        <PollState path="/dash/api/vitals" failure={failure} />
      ) : !sv ? (
        <StateView kind="inert" title="serving not reported by this server" detail="the running proxy predates vitals.serving; it appears after the next restart" />
      ) : (
        <div {...stylex.props(layout.stackSm)}>
          <div {...stylex.props(layout.grid2)}>
            <Stat label="ON THE CARD" value={sv.on_card ?? '—'} sub={sv.loaded === null ? 'llama-swap /running not read' : sv.on_card ? 'llama-swap /running' : 'neither main model loaded'} tone={sv.on_card && sv.on_card === sv.max ? 'rose' : 'moss'} />
            <Stat
              label="MAX TIER"
              value={sv.enabled && sv.max ? sv.max : sv.main}
              sub={sv.enabled ? `tier ${maxTier} · every other tier ${sv.main}` : `YAMADORI_MAX_MODEL unset · every tier ${sv.main}`}
              tone={sv.enabled ? 'rose' : undefined}
            />
          </div>
          {sv.enabled ? (
            <div {...stylex.props(layout.rowWrap)}>
              {Object.entries(sv.inflight ?? {}).map(([m, c]) => (
                <Chip key={m} tone={c > 0 ? 'cyan' : 'muted'}>
                  {m} · {c} IN FLIGHT
                </Chip>
              ))}
              {sv.switching_to ? <Chip tone="rose">SWITCHING TO {sv.switching_to}</Chip> : null}
              <Chip tone="muted">LAST MAX {sv.last_max_end ? `${ago(now - sv.last_max_end)} AGO` : '—'}</Chip>
              <Chip tone="muted">SWAP BACK {sv.idle_s == null ? 'ON THE NEXT NON-MAX REQUEST' : `AFTER ${n(sv.idle_s)} S IDLE`}</Chip>
            </div>
          ) : null}
          {sv.error ? <p {...stylex.props(text.labelXs, s.note)}>max_mode: {sv.error}</p> : null}
          {sv.loaded === null ? (
            <StateView kind="inert" title="llama-swap /running not read" />
          ) : !sv.loaded.length ? (
            <StateView kind="empty" title="llama-swap has nothing loaded" />
          ) : (
            <Table
              rows={sv.loaded}
              rowKey={(r: LoadedModel) => r.model}
              columns={[
                { key: 'm', head: 'loaded', cell: (r: LoadedModel) => (r.model === sv.on_card ? <span {...stylex.props(s.name)}>{r.model}</span> : r.model) },
                { key: 's', head: 'state', cell: (r: LoadedModel) => r.state },
                { key: 'p', head: 'port', num: true, cell: (r: LoadedModel) => r.port ?? '—' },
                { key: 'g', head: 'gguf', cell: (r: LoadedModel) => <span {...stylex.props(layout.truncate)}>{r.gguf ?? '—'}</span> },
              ]}
            />
          )}
        </div>
      )}
    </Panel>
  );
}

export function McpPanel({ fill }: P) {
  const p = usePoll<Mcp>(MCP_PATH, 30000);
  const d = p.data;
  const servers = Array.isArray(d?.servers) ? d.servers : [];
  const ready = servers.filter((x) => mcpState(x) === 'ready').length;
  return (
    <Panel
      kanji="繋"
      title="MCP · HOSTED SERVERS"
      tag={d ? (d.host?.enabled ? `${ready}/${servers.length} READY` : 'HOST OFF') : '—'}
      tagTone={d && d.host?.enabled ? 'moss' : 'muted'}
      flag="READ-ONLY"
      stale={p.stale}
      fill={fill}
    >
      {!d ? (
        <PollState path={MCP_PATH} failure={p.failure} />
      ) : (
        <div {...stylex.props(layout.stackSm)}>
          <p {...stylex.props(text.labelXs, s.note)}>
            {d.host?.enabled ? 'host on' : 'host off'} · {d.host?.switch ?? ''} · config {d.config?.exists ? d.config.path : 'built-in default (no servers file)'}
          </p>
          {!servers.length ? (
            <StateView kind="empty" title="no MCP server configured" />
          ) : (
            servers.map((x: McpServer) => {
              const st = mcpState(x);
              return (
                <div key={x.id} {...stylex.props(layout.stack)}>
                  <div {...stylex.props(layout.between)}>
                    <Label>
                      {x.title ?? x.id} · {x.package ?? '—'} · {x.runtime ?? '—'}
                      {x.network ? ` · ${x.network}` : ''}
                    </Label>
                    <Chip tone={STATE_TONE[st] ?? 'muted'} title={x.status?.why ?? undefined}>
                      {st.toUpperCase()}
                    </Chip>
                  </div>
                  {x.status?.why ? <p {...stylex.props(text.labelXs, s.note)}>{x.status.why}</p> : null}
                  <div {...stylex.props(layout.rowWrap)}>
                    {(x.tools ?? []).map((t) => (
                      <Chip key={t.name} tone="cyan" title={t.description}>
                        {t.name}
                        {t.upstream ? ` ← ${t.upstream}` : ''}
                      </Chip>
                    ))}
                  </div>
                  <p {...stylex.props(text.labelXs, s.tool)}>
                    {x.licence ? `licence ${x.licence}` : 'licence not recorded'}
                    {x.call_timeout_s != null ? ` · call timeout ${x.call_timeout_s} s` : ''}
                    {x.status?.server?.version ? ` · server ${x.status.server.name ?? ''} ${x.status.server.version}` : ''}
                    {x.status?.ready_since ? ` · ready since ${new Date(x.status.ready_since * 1000).toLocaleTimeString([], { hour12: false })}` : ''}
                  </p>
                </div>
              );
            })
          )}
        </div>
      )}
    </Panel>
  );
}
