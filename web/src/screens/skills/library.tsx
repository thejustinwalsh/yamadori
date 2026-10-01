// The Skills page's LIBRARY view, two more panels: the served skills by
// taxonomy area, and the held package indexes with what reads them. Both
// read /dash/api/skill-factory/library (mcp/dash_skills.py): what the
// retired NEBARI screen showed that still has a reader (operator, 2026-09-30).
import * as stylex from '@stylexjs/stylex';
import { AREA_AXES, areaRows, areasOf, errorOf, itemsOf, LIBRARY_PATH, packagesOf, toolsFor, type HeldPackage, type Library } from '../../api/library';
import type { Poll } from '../../api/usePoll';
import { n } from '../../format';
import { space } from '../../tokens/tokens.stylex';
import { Panel } from '../../ui/Panel';
import { Chip, Label, layout, Meter } from '../../ui/primitives';
import { PollState, StateView } from '../../ui/StateView';
import { Table } from '../../ui/Table';
import { text } from '../../ui/text';

const s = stylex.create({
  axes: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))' },
  bar: { display: 'grid', gridTemplateColumns: 'minmax(80px, 1fr) minmax(60px, 2fr) 36px', gap: space.spaceXs, alignItems: 'center' },
  name: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textTransform: 'none' },
  asIs: { textTransform: 'none', overflowWrap: 'anywhere' },
});

export function AreasPanel({ poll }: { poll: Poll<Library> }) {
  const a = areasOf(poll.data);
  return (
    <Panel kanji="域" title="LIBRARY BY AREA" tag={a ? `${n(a.served)} SERVED OF ${n(a.total)}` : '—'} tagTone="cyan" stale={poll.stale}
      sub="the SERVED skills (armed, enabled: what selection reads) counted along each taxonomy axis">
      {!poll.data ? (
        <PollState path={LIBRARY_PATH} failure={poll.failure} />
      ) : !a ? (
        <StateView kind="error" title={`${LIBRARY_PATH} · areas`} detail={errorOf(poll.data.areas) ?? undefined} />
      ) : !AREA_AXES.some((axis) => areaRows(a, axis).length) ? (
        <StateView kind="empty" title="no served skill is filed under an area" />
      ) : (
        <div {...stylex.props(s.axes)}>
          {AREA_AXES.map((axis) => {
            const rows = areaRows(a, axis);
            if (!rows.length) return null;
            const top = rows[0]?.n ?? 1;
            return (
              <div key={axis} {...stylex.props(layout.stack)}>
                <Label>{axis} · {rows.length}</Label>
                {rows.slice(0, 12).map((r) => (
                  <div key={r.id} {...stylex.props(s.bar)} title={`${r.label}: ${r.n} served skills`}>
                    <span {...stylex.props(text.labelXs, s.name)}>{r.label}</span>
                    <Meter value={r.n / Math.max(1, top)} tone={axis === 'framework' ? 'cyan' : 'moss'} label={`${r.label} ${r.n} served skills`} />
                    <span {...stylex.props(text.labelXs, text.num, text.primary)}>{n(r.n)}</span>
                  </div>
                ))}
                {rows.length > 12 ? <Label>+{rows.length - 12} more (the FACETS filter lists every one)</Label> : null}
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

export function PackagesPanel({ poll }: { poll: Poll<Library> }) {
  const pk = packagesOf(poll.data);
  const readers = poll.data?.readers ?? [];
  const ix = poll.data && poll.data.indexes && !('error' in poll.data.indexes) ? poll.data.indexes : null;
  return (
    <Panel kanji="蔵" title="HELD PACKAGES" tag={pk ? `${pk.length} INDEXES` : '—'} tagTone="cyan" stale={poll.stale}
      sub="index/packages: the package source the code-search MCP tools (tools API :1235) read, and the version the skills pipeline's PROVE type check installs">
      {!poll.data ? (
        <PollState path={LIBRARY_PATH} failure={poll.failure} />
      ) : !pk ? (
        <StateView kind="error" title={`${LIBRARY_PATH} · packages`} detail={errorOf(poll.data.packages) ?? undefined} />
      ) : !pk.length ? (
        <StateView kind="empty" title="no package index held (index/packages)" detail="python mcp/deps.py index name@version --embed" />
      ) : (
        <>
          <Table
            rows={pk}
            rowKey={(p: HeldPackage) => `${p.package}@${p.version}`}
            columns={[
              { key: 'p', head: 'package', cell: (p: HeldPackage) => <span {...stylex.props(s.asIs)}>{p.package}@{p.version}</span> },
              {
                key: 'e',
                head: 'index',
                cell: (p: HeldPackage) => (
                  <span {...stylex.props(layout.rowWrap)}>
                    {p.embedded ? <Chip tone="moss">EMBEDDED</Chip> : <Chip tone="muted">NOT EMBEDDED</Chip>}
                    {p.complete === false ? <Chip tone="rose">INCOMPLETE</Chip> : null}
                  </span>
                ),
              },
              { key: 'i', head: 'items', num: true, cell: (p: HeldPackage) => n(itemsOf(p)) },
              { key: 'd', head: 'defs', num: true, cell: (p: HeldPackage) => n(p.defs) },
              { key: 'f', head: 'files', num: true, cell: (p: HeldPackage) => n(p.files) },
              { key: 'pub', head: 'published', cell: (p: HeldPackage) => p.published ?? '—' },
              { key: 'u', head: 'read by', cell: (p: HeldPackage) => <span {...stylex.props(text.labelXs, s.asIs)}>{toolsFor(p, readers).join(' · ')}</span> },
            ]}
          />
          {readers.map((r) => (
            <p key={r.who} {...stylex.props(text.labelXs, text.muted)} style={{ margin: 0 }}>
              {r.who}: {r.tools.join(', ')}
              {r.needs_embedding.length ? ` (${r.needs_embedding.join(', ')} needs the index embedded)` : ''} · {r.source}
            </p>
          ))}
          {ix ? (
            <Label>
              bound code index {n(ix.code ? ix.code.chunks + ix.code.defs : null)} items · {n(ix.repos?.indexes ?? 0)} repository indexes
            </Label>
          ) : null}
        </>
      )}
    </Panel>
  );
}
