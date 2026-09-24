import * as stylex from '@stylexjs/stylex';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useShared } from '../api/data';
import type { Vitals } from '../api/types';
import { ago, gib, n } from '../format';
import { href, type Route } from '../router';
import { colors, space } from '../tokens/tokens.stylex';
import { NavLink } from './NavLink';
import { Chip, Dot, Label, Meter } from './primitives';
import { text } from './text';

const MOBILE = '@media (max-width: 767px)';
// One row (brand, screens, status) only where it fits; tablets put the screens
// on a second row inside the same sticky header.
const WIDE = '@media (min-width: 1200px)';

const s = stylex.create({
  app: {
    minHeight: '100vh',
    backgroundColor: colors.surfaceContainerLowest,
    color: colors.onSurface,
    // A faint blueprint grid behind everything, like the mockups' substrate.
    backgroundImage: `linear-gradient(color-mix(in srgb, ${colors.outlineVariant} 12%, transparent) 1px, transparent 1px), linear-gradient(90deg, color-mix(in srgb, ${colors.outlineVariant} 12%, transparent) 1px, transparent 1px)`,
    backgroundSize: '32px 32px',
    // Room under the last panel for the floating screens button.
    paddingBottom: { default: 0, [MOBILE]: '80px' },
    overflowX: 'clip',
  },
  header: {
    position: 'sticky',
    top: 0,
    zIndex: 50,
    // Translucent: the page shows through, blurred, as it scrolls beneath.
    backgroundColor: `color-mix(in srgb, ${colors.surfaceContainerLowest} 68%, transparent)`,
    backdropFilter: 'blur(14px) saturate(140%)',
    WebkitBackdropFilter: 'blur(14px) saturate(140%)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: `color-mix(in srgb, ${colors.outlineVariant} 35%, transparent)`,
  },
  headInner: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr) auto', [WIDE]: 'minmax(0, 1fr) auto auto' },
    gridTemplateAreas: { default: '"brand right" "nav nav"', [MOBILE]: '"brand right"', [WIDE]: '"brand nav right"' },
    alignItems: 'center',
    columnGap: space.spaceMd,
    maxWidth: '1440px',
    marginInline: 'auto',
    paddingInline: space.margin,
  },
  brand: { gridArea: 'brand', display: 'flex', alignItems: 'center', gap: space.spaceSm, minWidth: 0, paddingBlock: space.spaceXs },
  brandText: { minWidth: 0 },
  brandRow: { display: 'flex', alignItems: 'center', gap: space.spaceSm, minWidth: 0 },
  glyph: {
    color: colors.primaryContainer,
    textShadow: `0 0 10px ${colors.primaryContainer}`,
    fontSize: '20px',
    lineHeight: 1,
  },
  name: { color: colors.primary, letterSpacing: '0.08em', margin: 0, whiteSpace: 'nowrap' },
  modelChip: { minWidth: 0, overflow: 'hidden', display: { default: 'inline-flex', [MOBILE]: 'none' } },
  subline: {
    color: colors.onSurfaceVariant,
    margin: 0,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  right: { gridArea: 'right', display: 'flex', alignItems: 'center', gap: space.spaceSm, paddingBlock: space.spaceXs },
  pod: {
    display: { default: 'flex', [MOBILE]: 'none' },
    flexDirection: 'column',
    gap: '3px',
    minWidth: '150px',
    paddingInline: space.spaceSm,
    paddingBlock: '4px',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
    backgroundColor: `color-mix(in srgb, ${colors.surfaceContainerLow} 80%, transparent)`,
  },
  podRow: { display: 'flex', justifyContent: 'space-between', gap: space.spaceSm, alignItems: 'baseline' },
  podVal: { color: colors.primaryContainer },
  live: { display: 'flex', alignItems: 'center', gap: '6px', color: colors.onSurfaceVariant, whiteSpace: 'nowrap' },
  nav: {
    gridArea: 'nav',
    display: { default: 'flex', [MOBILE]: 'none' },
    alignSelf: 'stretch',
    gap: 0,
    marginInline: { default: `calc(-1 * ${space.spaceMd})`, [WIDE]: 0 },
  },
  tab: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    paddingInline: space.spaceMd,
    paddingBlock: space.spaceSm,
    color: colors.outline,
    textDecoration: 'none',
    whiteSpace: 'nowrap',
    borderBottomWidth: 2,
    borderBottomStyle: 'solid',
    borderBottomColor: 'transparent',
    cursor: 'pointer',
    outlineOffset: '-2px',
    ':hover': { color: colors.onSurface },
  },
  tabOn: {
    color: colors.primaryContainer,
    borderBottomColor: colors.primaryContainer,
    textShadow: `0 0 12px color-mix(in srgb, ${colors.primaryContainer} 50%, transparent)`,
  },
  tabIdx: { color: colors.outlineVariant, display: { default: 'inline', '@media (min-width: 1200px) and (max-width: 1439px)': 'none' } },
  main: {
    maxWidth: '1440px',
    marginInline: 'auto',
    padding: space.margin,
    display: 'flex',
    flexDirection: 'column',
    gap: space.gutter,
  },
  footer: {
    maxWidth: '1440px',
    marginInline: 'auto',
    paddingInline: space.margin,
    paddingBottom: space.spaceMd,
    display: 'flex',
    flexWrap: 'wrap',
    gap: space.spaceMd,
    color: colors.outline,
  },
  // Phones: the screens collapse into one floating button.
  fab: {
    display: { default: 'none', [MOBILE]: 'flex' },
    position: 'fixed',
    right: space.margin,
    bottom: `calc(${space.margin} + env(safe-area-inset-bottom, 0px))`,
    zIndex: 60,
    width: '52px',
    height: '52px',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    cursor: 'pointer',
    color: colors.primaryContainer,
    backgroundColor: `color-mix(in srgb, ${colors.surfaceContainerLow} 78%, transparent)`,
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.primaryContainer} 55%, transparent)`,
    borderRadius: 0,
    boxShadow: `0 0 18px -4px color-mix(in srgb, ${colors.primaryContainer} 45%, transparent)`,
  },
  fabOpen: { backgroundColor: `color-mix(in srgb, ${colors.primaryContainer} 16%, ${colors.surfaceContainerLow})` },
  scrim: {
    display: { default: 'none', [MOBILE]: 'block' },
    position: 'fixed',
    inset: 0,
    zIndex: 55,
    backgroundColor: `color-mix(in srgb, ${colors.surfaceContainerLowest} 45%, transparent)`,
  },
  menu: {
    display: { default: 'none', [MOBILE]: 'flex' },
    flexDirection: 'column',
    position: 'fixed',
    right: space.margin,
    bottom: `calc(${space.margin} + 60px + env(safe-area-inset-bottom, 0px))`,
    zIndex: 60,
    width: `min(260px, calc(100vw - 2 * ${space.margin}))`,
    boxSizing: 'border-box',
    margin: 0,
    padding: space.spaceXs,
    listStyle: 'none',
    backgroundColor: `color-mix(in srgb, ${colors.surfaceContainerLow} 86%, transparent)`,
    backdropFilter: 'blur(14px) saturate(140%)',
    WebkitBackdropFilter: 'blur(14px) saturate(140%)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 60%, transparent)`,
  },
  menuItem: {
    display: 'grid',
    gridTemplateColumns: '28px 56px minmax(0, 1fr)',
    alignItems: 'center',
    gap: space.spaceSm,
    paddingInline: space.spaceSm,
    paddingBlock: space.spaceSm,
    minHeight: '44px',
    boxSizing: 'border-box',
    color: colors.onSurfaceVariant,
    textDecoration: 'none',
    borderInlineStartWidth: 2,
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: 'transparent',
  },
  menuOn: {
    color: colors.primaryContainer,
    borderInlineStartColor: colors.primaryContainer,
    backgroundColor: `color-mix(in srgb, ${colors.primaryContainer} 8%, transparent)`,
  },
});

const TABS: { route: Route['name']; kanji: string; label: string; to: string; idx: string }[] = [
  { route: 'tokonoma', kanji: '床の間', label: 'TOKONOMA', to: href.tokonoma, idx: '01' },
  { route: 'nebari', kanji: '根張り', label: 'NEBARI', to: href.nebari, idx: '02' },
  { route: 'naedoko', kanji: '苗床', label: 'NAEDOKO', to: href.naedoko, idx: '03' },
  { route: 'sentei', kanji: '剪定', label: 'SENTEI', to: href.sentei, idx: '04' },
  { route: 'settings', kanji: '設定', label: 'SETTINGS', to: href.settings, idx: '05' },
  { route: 'skills', kanji: '技', label: 'SKILLS', to: href.skills, idx: '06' },
];

/** The model actually being served: the gguf on the bonsai listener's port. */
export function servedModel(v: Vitals | null): string | null {
  if (!v) return null;
  const port = v.listeners?.find((l) => l.role === 'bonsai')?.port;
  const p = port ? v.processes?.find((x) => String(x.port) === String(port)) : null;
  return p?.what ? p.what.replace(/\.gguf$/i, '') : null;
}

function Header({ current }: { current: Route['name'] }) {
  const { vitals } = useShared();
  const v = vitals.data;
  const used = v?.gpus?.reduce((a, g) => a + g.used_mib, 0) ?? null;
  const total = v?.gpus?.reduce((a, g) => a + g.total_mib, 0) ?? null;
  const ctx = v?.context && 'pool' in v.context ? v.context : null;
  const model = servedModel(v);
  const dotState = vitals.failure && !v ? 'down' : !v ? 'dead' : vitals.stale ? 'stale' : 'live';
  const sub = v
    ? [v.gpus?.map((g) => `GPU${g.index} ${g.name.replace(/^NVIDIA (GeForce )?/, '')}`).join(' + ') || 'no GPU reported', ctx ? `-c ${n(ctx.pool)}` : null, ctx ? `KV ${ctx.gib} GiB` : null]
        .filter(Boolean)
        .join(' · ')
    : vitals.failure
      ? 'vitals unavailable'
      : 'reading /dash/api/vitals…';
  return (
    <header {...stylex.props(s.header)}>
      <div {...stylex.props(s.headInner)}>
        <div {...stylex.props(s.brand)}>
          <span aria-hidden {...stylex.props(s.glyph)}>▟</span>
          <div {...stylex.props(s.brandText)}>
            <div {...stylex.props(s.brandRow)}>
              <h1 {...stylex.props(text.headlineSm, s.name)}>YAMADORI</h1>
              <span {...stylex.props(s.modelChip)}>
                <Chip tone="moss" title="the gguf served on the bonsai listener, from /dash/api/vitals">
                  <span {...stylex.props(text.kanji)}>山採り</span> · {model ?? (v ? 'no model on the bonsai port' : '…')}
                </Chip>
              </span>
            </div>
            <p {...stylex.props(text.labelXs, s.subline)}>{sub}</p>
          </div>
        </div>
        <nav aria-label="screens" {...stylex.props(s.nav)}>
          {TABS.map((t) => (
            <NavLink
              key={t.route}
              to={t.to}
              aria-current={current === t.route ? 'page' : undefined}
              {...stylex.props(text.labelMd, s.tab, current === t.route && s.tabOn)}
            >
              <span {...stylex.props(s.tabIdx)}>#{t.idx}</span>
              <span {...stylex.props(text.kanji)}>{t.kanji}</span>
              {t.label}
            </NavLink>
          ))}
        </nav>
        <div {...stylex.props(s.right)}>
          <div {...stylex.props(s.pod)}>
            <div {...stylex.props(s.podRow)}>
              <Label>VRAM · {v ? `${v.gpus?.length ?? 0} GPU` : "—"}</Label>
              <span {...stylex.props(text.labelMd, s.podVal, text.num)}>
                {used !== null && total ? `${gib(used)} / ${gib(total)} GiB` : '—'}
              </span>
            </div>
            <Meter value={used !== null && total ? used / total : null} label="VRAM used across GPUs" />
          </div>
          <span {...stylex.props(text.labelXs, s.live)} title="age of the vitals snapshot">
            <Dot state={dotState} />
            {vitals.age !== null ? `${vitals.stale ? 'STALE ' : ''}${ago(vitals.age)}` : vitals.failure ? 'NO DATA' : '…'}
          </span>
        </div>
      </div>
    </header>
  );
}

function RoutesIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M4 6h14M4 11h14M4 16h9" />
      <rect x="15.5" y="14.5" width="3" height="3" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Phones: the screens as one floating button that opens a list. */
function RoutesFab({ current }: { current: Route['name'] }) {
  const [open, setOpen] = useState(false);
  const menu = useRef<HTMLUListElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    // React's autoFocus does not apply to links: focus the current screen.
    (menu.current?.querySelector<HTMLElement>('[aria-current="page"]') ?? menu.current?.querySelector<HTMLElement>('a'))?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        button.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);
  return (
    <>
      {open && <div aria-hidden {...stylex.props(s.scrim)} onClick={() => setOpen(false)} />}
      {open && (
        <ul ref={menu} id="screens-menu" aria-label="screens" {...stylex.props(s.menu)}>
          {TABS.map((t) => (
            <li key={t.route}>
              <NavLink
                to={t.to}
                onNavigate={() => setOpen(false)}
                aria-current={current === t.route ? 'page' : undefined}
                {...stylex.props(text.labelMd, s.menuItem, current === t.route && s.menuOn)}
              >
                <span {...stylex.props(s.tabIdx)}>#{t.idx}</span>
                <span {...stylex.props(text.kanji)}>{t.kanji}</span>
                {t.label}
              </NavLink>
            </li>
          ))}
        </ul>
      )}
      <button
        ref={button}
        type="button"
        aria-label="screens"
        aria-expanded={open}
        aria-controls="screens-menu"
        onClick={() => setOpen((o) => !o)}
        {...stylex.props(s.fab, open && s.fabOpen)}
      >
        <RoutesIcon />
      </button>
    </>
  );
}

export function Shell({ route, children }: { route: Route; children: ReactNode }) {
  const current = route.name === 'dataset' ? 'naedoko' : route.name;
  const { vitals } = useShared();
  return (
    <div {...stylex.props(s.app)}>
      <Header current={current} />
      <main {...stylex.props(s.main)}>{children}</main>
      <footer {...stylex.props(text.labelXs, s.footer)}>
        <span>SNAPSHOT {vitals.data ? new Date(vitals.data.at * 1000).toLocaleString([], { hour12: false }) : '—'}</span>
        <span>POLL 5s · STALE AFTER 30s</span>
      </footer>
      <RoutesFab current={current} />
    </div>
  );
}
