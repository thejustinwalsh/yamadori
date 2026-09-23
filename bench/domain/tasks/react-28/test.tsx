import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { Tooltip } from './solution';
import { advance } from './helpers';

// fireEvent/act rather than userEvent: RTL's async wrapper only advances
// *jest* fake timers, so userEvent under vitest fake timers never settles.

const button = () => screen.getByRole('button', { name: 'Save' });
const tip = () => screen.queryByRole('tooltip');

function expectShown(content: string) {
  const t = tip();
  expect(t).not.toBeNull();
  expect(t!.textContent).toBe(content);
  expect(t!.id).not.toBe('');
  expect(button().getAttribute('aria-describedby')).toBe(t!.id);
}

function expectHidden() {
  expect(tip()).toBeNull();
  expect(button().getAttribute('aria-describedby')).toBeNull();
}

const enter = () => fireEvent.mouseEnter(button());
const leave = () => fireEvent.mouseLeave(button());
const focus = () => act(() => button().focus());
const blur = () => act(() => button().blur());

describe('Tooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('renders the button with no tooltip at first', () => {
    render(<Tooltip content="Save your work" label="Save" />);
    expect(button().textContent).toBe('Save');
    expectHidden();
  });

  it('shows on hover after the default 300 ms and links it with aria-describedby', () => {
    render(<Tooltip content="Save your work" label="Save" />);
    enter();
    advance(299);
    expectHidden();
    advance(1);
    expectShown('Save your work');
  });

  it('honours a custom delay', () => {
    render(<Tooltip content="Ctrl+S" label="Save" delay={1000} />);
    enter();
    advance(999);
    expectHidden();
    advance(1);
    expectShown('Ctrl+S');
  });

  it('hides immediately on mouseleave', () => {
    render(<Tooltip content="Save your work" label="Save" />);
    enter();
    advance(300);
    expectShown('Save your work');
    leave();
    expectHidden();
  });

  it('never appears when the mouse leaves before the delay, and a new visit starts a fresh delay', () => {
    render(<Tooltip content="Save your work" label="Save" />);
    enter();
    advance(200);
    leave();
    advance(1000);
    expectHidden();
    enter();
    advance(200);
    expectHidden();
    advance(100);
    expectShown('Save your work');
  });

  it('shows on focus after the delay and hides on blur', () => {
    render(<Tooltip content="Save your work" label="Save" delay={500} />);
    focus();
    advance(499);
    expectHidden();
    advance(1);
    expectShown('Save your work');
    blur();
    expectHidden();
    focus();
    advance(100);
    blur();
    advance(1000);
    expectHidden();
  });

  it('hides on Escape while the button has focus', () => {
    render(<Tooltip content="Save your work" label="Save" />);
    focus();
    advance(300);
    expectShown('Save your work');
    fireEvent.keyDown(button(), { key: 'Escape' });
    expectHidden();
  });

  it('leaves no timer pending after unmount', () => {
    const { unmount } = render(<Tooltip content="Save your work" label="Save" />);
    enter();
    advance(100);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
