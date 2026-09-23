import { describe, it, expect, vi } from 'vitest';
import { useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Modal } from './solution';

function Harness({ onClose, empty = false }: { onClose?: () => void; empty?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)}>Open settings</button>
      <button>Outside</button>
      <Modal
        open={open}
        title="Settings"
        onClose={() => {
          onClose?.();
          setOpen(false);
        }}
      >
        {empty ? (
          <p>Nothing to do here.</p>
        ) : (
          <>
            <input aria-label="Name" />
            <button disabled>Disabled</button>
            <button>Save</button>
            <button onClick={() => setOpen(false)}>Cancel</button>
          </>
        )}
      </Modal>
      <button>After</button>
    </>
  );
}

async function openIt(props: { onClose?: () => void; empty?: boolean } = {}) {
  const user = userEvent.setup({ delay: null });
  render(<Harness {...props} />);
  const opener = screen.getByRole('button', { name: 'Open settings' });
  await user.click(opener);
  return { user, opener };
}

const focused = () => document.activeElement;
const byName = (name: string) => screen.getByRole('button', { name });

describe('Modal', () => {
  it('renders nothing while closed', () => {
    render(
      <Modal open={false} onClose={() => {}} title="Hidden">
        <button>Inside</button>
      </Modal>,
    );
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.queryByText('Hidden')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Inside' })).toBeNull();
  });

  it('renders a labelled modal dialog while open', async () => {
    await openIt();
    const dialog = screen.getByRole('dialog', { name: 'Settings' });
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    const heading = document.getElementById(dialog.getAttribute('aria-labelledby') ?? '');
    expect(heading?.textContent).toBe('Settings');
    expect(dialog.contains(heading)).toBe(true);
  });

  it('moves focus to the first focusable element on open', async () => {
    await openIt();
    expect(focused()).toBe(screen.getByLabelText('Name'));
  });

  it('focuses the dialog itself when it holds nothing focusable', async () => {
    await openIt({ empty: true });
    expect(focused()).toBe(screen.getByRole('dialog'));
  });

  it('Tab and Shift+Tab cycle inside the dialog', async () => {
    const { user } = await openIt();
    const name = screen.getByLabelText('Name');
    await user.tab();
    expect(focused()).toBe(byName('Save'));
    await user.tab();
    expect(focused()).toBe(byName('Cancel'));
    await user.tab();
    expect(focused()).toBe(name);
    await user.tab({ shift: true });
    expect(focused()).toBe(byName('Cancel'));
    await user.tab({ shift: true });
    expect(focused()).toBe(byName('Save'));
    for (let i = 0; i < 7; i++) {
      await user.tab();
      expect(screen.getByRole('dialog').contains(focused())).toBe(true);
    }
  });

  it('Escape calls onClose once and focus returns to the opener', async () => {
    const onClose = vi.fn();
    const { user, opener } = await openIt({ onClose });
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(focused()).toBe(opener);
  });

  it('returns focus when the parent closes it by other means', async () => {
    const { user, opener } = await openIt();
    await user.click(byName('Cancel'));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(focused()).toBe(opener);
  });

  it('Escape does nothing after the dialog has closed', async () => {
    const onClose = vi.fn();
    const { user } = await openIt({ onClose });
    await user.click(byName('Cancel'));
    await user.keyboard('{Escape}');
    expect(onClose).not.toHaveBeenCalled();
    await user.click(byName('Open settings'));
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
