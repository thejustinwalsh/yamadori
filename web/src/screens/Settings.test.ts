// Static renders of the image-model picker (vitest runs in node, so these
// render to markup with react-dom/server), and the pure helpers behind it.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { choiceBody, secondsText, type ImageModelOption, type ImageSettings } from '../api/settings';
import { ImageModelPicker } from './Settings';

const base: ImageModelOption = {
  id: 'base',
  name: 'yamadori-image',
  label: 'Qwen-Image-2.1',
  steps: 20,
  est_seconds: 107,
  est_basis: 'measured, n=10, 105-111 s',
  licence: 'Qwen Research License (non-commercial)',
};
const turbo: ImageModelOption = {
  id: 'turbo',
  name: 'yamadori-image-turbo',
  label: 'Qwen-Image-2.1 turbo (Viggle, 4-step)',
  steps: 4,
  est_seconds: 24,
  est_basis: 'n=1 smoke test, sd-cli',
  licence: 'Qwen Research License (non-commercial)',
};
const settings = (over: Partial<ImageSettings>): ImageSettings => ({
  ok: true,
  choice: null,
  default: 'base',
  effective: 'base',
  options: [base, turbo],
  ...over,
});
const checkedRe = (id: string) => new RegExp(`<input[^>]*checked=""[^>]*value="${id}"`);
const render = (data: ImageSettings, busy = false) =>
  renderToStaticMarkup(createElement(ImageModelPicker, { data, busy, onChoose: () => {} }));

describe('ImageModelPicker', () => {
  it('lists every option with its steps, seconds and licence', () => {
    const html = render(settings({}));
    expect(html).toContain('Qwen-Image-2.1 turbo (Viggle, 4-step)');
    expect(html).toContain('>20<');
    expect(html).toContain('>4<');
    expect(html).toContain('~107 s · measured, n=10, 105-111 s');
    expect(html).toContain('~24 s · n=1 smoke test, sd-cli');
    expect(html.match(/Qwen Research License \(non-commercial\)/g)?.length).toBe(2);
    expect(html).toContain('yamadori-image-turbo');
  });
  it('checks the effective model; with no choice it is the default and says so', () => {
    const html = render(settings({}));
    expect(html).toMatch(checkedRe('base'));
    expect(html).not.toMatch(checkedRe('turbo'));
    expect(html).toContain('IN USE · DEFAULT');
    expect(html).toContain('SERVER DEFAULT');
    expect(html).not.toContain('[ USE SERVER DEFAULT ]');
  });
  it('a saved choice is checked, and offers the way back to the default', () => {
    const html = render(settings({ choice: 'turbo', effective: 'turbo' }));
    expect(html).toMatch(checkedRe('turbo'));
    expect(html).not.toMatch(checkedRe('base'));
    expect(html).toContain('>IN USE<');
    expect(html).toContain('[ USE SERVER DEFAULT ]');
  });
  it('marks whichever model the server default is', () => {
    const html = render(settings({ default: 'turbo', effective: 'turbo' }));
    expect(html).toMatch(/data-option="turbo"[^]*SERVER DEFAULT/);
    expect(html).not.toMatch(/data-option="base"[^]*SERVER DEFAULT[^]*data-option="turbo"/);
  });
  it('is disabled while a save is in flight', () => {
    expect(render(settings({}), true)).toMatch(/<fieldset[^>]*disabled=""/);
  });
});

describe('settings helpers', () => {
  it('an unmeasured option says so instead of inventing a number', () => {
    expect(secondsText({ ...turbo, est_seconds: null, est_basis: 'not measured' })).toBe('not measured');
  });
  it('a pick sends {choice}, and null clears it', () => {
    expect(choiceBody('turbo')).toEqual({ choice: 'turbo' });
    expect(choiceBody(null)).toEqual({ choice: null });
  });
});
