// A link to a screen that loads the screen before it is shown. Hover, focus
// and press start the chunk and the data (src/routes.ts); a click waits up to
// NAV_WAIT_MS for whatever is still missing, then navigates either way, so a
// slow endpoint costs at most that long and the screen shows its own state.
import { useState, type AnchorHTMLAttributes, type MouseEvent, type ReactNode } from 'react';
import { useLocation, useRouter } from 'wouter';
import { resolve } from '../router';
import { preloadRoute } from '../routes';

const NAV_WAIT_MS = 600;

type Props = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href' | 'onClick'> & {
  to: string;
  /** after the click is accepted, before navigation (closes a menu) */
  onNavigate?: () => void;
  children: ReactNode;
};

const wait = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

export function NavLink({ to, onNavigate, children, ...rest }: Props) {
  const router = useRouter();
  const [, go] = useLocation();
  const [pending, setPending] = useState(false);

  const intent = () => {
    void preloadRoute(resolve(router.parser, to)).done;
  };

  const onClick = async (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    const p = preloadRoute(resolve(router.parser, to));
    if (!p.ready) {
      setPending(true);
      await Promise.race([p.done, wait(NAV_WAIT_MS)]);
      setPending(false);
    }
    onNavigate?.();
    if (to !== window.location.pathname) {
      go(to);
      window.scrollTo(0, 0);
    }
  };

  return (
    <a
      {...rest}
      href={to}
      onPointerEnter={intent}
      onFocus={intent}
      onPointerDown={intent}
      onClick={onClick}
      aria-busy={pending || undefined}
    >
      {children}
    </a>
  );
}
