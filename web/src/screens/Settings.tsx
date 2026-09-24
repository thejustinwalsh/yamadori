// 設定 SETTINGS — this key's own choices. One today: which image model draws
// for it (GET/PUT /dash/api/settings/image). The key decides whose account
// is written; there is no account field to change.
import * as stylex from '@stylexjs/stylex';
import { useState } from 'react';
import { putJson } from '../api/client';
import { choiceBody, IMAGE_SETTINGS_PATH, secondsText, type ImageSettings } from '../api/settings';
import { usePoll } from '../api/usePoll';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout } from '../ui/primitives';
import { PollState } from '../ui/StateView';
import { text } from '../ui/text';

const s = stylex.create({
  list: { display: 'grid', gap: space.spaceXs, gridTemplateColumns: 'minmax(0, 1fr)', margin: 0, padding: 0, borderWidth: 0, borderStyle: 'none', minWidth: 0 },
  option: {
    display: 'grid',
    gridTemplateColumns: '20px minmax(0, 1fr)',
    columnGap: space.spaceSm,
    rowGap: space.spaceXs,
    alignItems: 'start',
    padding: space.spaceSm,
    minHeight: '44px',
    boxSizing: 'border-box',
    cursor: 'pointer',
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
    ':hover': { borderColor: `color-mix(in srgb, ${colors.primaryContainer} 45%, transparent)` },
  },
  optionOn: {
    borderColor: `color-mix(in srgb, ${colors.primaryContainer} 70%, transparent)`,
    boxShadow: `inset 2px 0 0 ${colors.primaryContainer}`,
  },
  radio: { margin: 0, marginTop: '3px', accentColor: colors.primaryContainer, width: '16px', height: '16px' },
  body: { display: 'flex', flexDirection: 'column', gap: space.spaceXs, minWidth: 0 },
  head: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: space.spaceXs },
  name: { color: colors.primary, overflowWrap: 'anywhere' },
  facts: {
    display: 'grid',
    gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', '@media (max-width: 520px)': 'minmax(0, 1fr)' },
    gap: space.spaceXs,
    margin: 0,
  },
  fact: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 },
  dd: { margin: 0, color: colors.onSurface, overflowWrap: 'anywhere', textTransform: 'none' },
  button: {
    backgroundColor: { default: 'transparent', ':hover': `color-mix(in srgb, ${colors.tertiaryContainer} 10%, transparent)` },
    color: colors.tertiaryContainer,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.outlineVariant,
    paddingInline: space.spaceMd,
    paddingBlock: space.spaceXs,
    minHeight: '32px',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { default: 1, ':disabled': 0.5 },
    whiteSpace: 'nowrap',
  },
  note: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere' },
  // The API model name, as a client sends it: not uppercased.
  apiName: { color: colors.outline, textTransform: 'none', overflowWrap: 'anywhere' },
  warn: { margin: 0, color: colors.secondary, overflowWrap: 'anywhere' },
});

type Msg = { tone: 'busy' | 'ok' | 'err'; text: string } | null;

/** The picker, drawn from one settings snapshot. Presentational: `onChoose`
 * does the saving. `null` means "follow the server default". */
export function ImageModelPicker({
  data,
  busy = false,
  msg = null,
  onChoose,
}: {
  data: ImageSettings;
  busy?: boolean;
  msg?: Msg;
  onChoose: (id: string | null) => void;
}) {
  const current = data.effective;
  return (
    <div {...stylex.props(layout.stackSm)}>
      <fieldset aria-label="image model" disabled={busy} {...stylex.props(s.list)}>
        {data.options.map((o) => {
          const on = o.id === current;
          return (
            <label key={o.id} data-option={o.id} {...stylex.props(s.option, on && s.optionOn)}>
              <input
                type="radio"
                name="image-model"
                value={o.id}
                checked={on}
                onChange={() => onChoose(o.id)}
                {...stylex.props(s.radio)}
              />
              <span {...stylex.props(s.body)}>
                <span {...stylex.props(s.head)}>
                  <span {...stylex.props(text.labelMd, s.name)}>{o.label}</span>
                  {o.id === data.default ? <Chip tone="muted">SERVER DEFAULT</Chip> : null}
                  {on ? <Chip tone="moss">{data.choice === null ? 'IN USE · DEFAULT' : 'IN USE'}</Chip> : null}
                </span>
                <dl {...stylex.props(s.facts)}>
                  <div {...stylex.props(s.fact)}>
                    <dt>
                      <Label>STEPS</Label>
                    </dt>
                    <dd {...stylex.props(text.bodySm, text.num, s.dd)}>{o.steps}</dd>
                  </div>
                  <div {...stylex.props(s.fact)}>
                    <dt>
                      <Label>SECONDS / 1024²</Label>
                    </dt>
                    <dd {...stylex.props(text.bodySm, s.dd)}>{secondsText(o)}</dd>
                  </div>
                  <div {...stylex.props(s.fact)}>
                    <dt>
                      <Label>LICENCE</Label>
                    </dt>
                    <dd {...stylex.props(text.bodySm, s.dd)}>{o.licence}</dd>
                  </div>
                </dl>
                <span {...stylex.props(text.labelXs, s.apiName)}>{o.name}</span>
              </span>
            </label>
          );
        })}
      </fieldset>
      <div {...stylex.props(layout.rowWrap)}>
        {data.choice !== null ? (
          <button type="button" disabled={busy} onClick={() => onChoose(null)} {...stylex.props(text.labelMd, s.button)}>
            [ USE SERVER DEFAULT ]
          </button>
        ) : null}
        {msg ? (
          <p role={msg.tone === 'err' ? 'alert' : 'status'} {...stylex.props(text.bodySm, msg.tone === 'err' ? s.warn : s.note)}>
            {msg.tone === 'err' ? '[ ! ] ' : ''}
            {msg.text}
          </p>
        ) : null}
      </div>
    </div>
  );
}

function ImagePanel() {
  const poll = usePoll<ImageSettings>(IMAGE_SETTINGS_PATH, 0);
  const [saved, setSaved] = useState<ImageSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const data = saved ?? poll.data;

  const choose = async (id: string | null) => {
    setBusy(true);
    setMsg({ tone: 'busy', text: 'saving' });
    const r = await putJson<ImageSettings>(IMAGE_SETTINGS_PATH, choiceBody(id));
    setBusy(false);
    if (!r.ok) {
      setMsg({ tone: 'err', text: `not saved: ${'message' in r.failure ? r.failure.message : r.failure.kind}` });
      return;
    }
    setSaved(r.data);
    setMsg({ tone: 'ok', text: 'saved' });
    poll.reload();
  };

  return (
    <Panel kanji="画" title="IMAGE MODEL" tag="THIS KEY" tagTone="cyan" edge="moss" fill>
      {data ? <ImageModelPicker data={data} busy={busy} msg={msg} onChoose={(id) => void choose(id)} /> : <PollState path={IMAGE_SETTINGS_PATH} failure={poll.failure} />}
    </Panel>
  );
}

export function Settings() {
  return (
    <Bento areas={areas.settings}>
      <Cell area="image">
        <ErrorBoundary what="IMAGE MODEL" source={IMAGE_SETTINGS_PATH} fill>
          <ImagePanel />
        </ErrorBoundary>
      </Cell>
    </Bento>
  );
}

const areas = stylex.create({
  settings: {
    gridTemplateAreas: {
      default: '"image"',
      [MQ.tablet]: '"image image image image image image"',
      [MQ.desktop]: '"image image image image image image image . . . . ."',
    },
  },
});
