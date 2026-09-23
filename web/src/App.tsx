import * as stylex from '@stylexjs/stylex';
import { Suspense, type ReactNode } from 'react';
import { DataProvider, useShared } from './api/data';
import { useKey } from './api/usePoll';
import { useRoute } from './router';
import { screens } from './routes';
import { Cockpit } from './screens/Cockpit';
import { colors } from './tokens/tokens.stylex';
import { Shell } from './ui/Shell';
import { SignIn } from './ui/SignIn';
import { ErrorBoundary } from './ui/ErrorBoundary';
import { StateView } from './ui/StateView';

// Lazy screens that nav links preload on intent (src/routes.ts).
const Phase0 = screens.phase0.Component;
const Nebari = screens.nebari.Component;
const Naedoko = screens.naedoko.Component;
const DatasetDetail = screens.dataset.Component;
const Sentei = screens.sentei.Component;

// The page background is set on <body> from a token, so there is no white
// flash and no hex outside src/tokens.
const bodyStyle = stylex.create({ body: { margin: 0, backgroundColor: colors.surfaceContainerLowest, color: colors.onSurface } });
if (typeof document !== 'undefined') {
  document.body.className = stylex.props(bodyStyle.body).className ?? '';
}

function Gate({ children }: { children: ReactNode }) {
  const key = useKey();
  const { vitals } = useShared();
  const f = vitals.failure;
  if (!key || f?.kind === 'unauthorised' || f?.kind === 'nokey') return <SignIn failure={f} />;
  return <>{children}</>;
}

function Screen() {
  const route = useRoute();
  const body = (() => {
    switch (route.name) {
      case 'tokonoma':
        return <Cockpit />;
      case 'nebari':
        return <Nebari />;
      case 'naedoko':
        return <Naedoko />;
      case 'dataset':
        return <DatasetDetail id={route.id} />;
      case 'sentei':
        return <Sentei />;
      case 'notfound':
        return <StateView kind="empty" title={`no screen at ${route.path}`} />;
      default:
        return null;
    }
  })();
  return (
    <Shell route={route}>
      <Gate>
        <ErrorBoundary what={route.name.toUpperCase()} bare>
          <Suspense fallback={<StateView kind="loading" title="loading screen" />}>{body}</Suspense>
        </ErrorBoundary>
      </Gate>
    </Shell>
  );
}

export function App() {
  const route = useRoute();
  if (route.name === 'phase0') {
    return (
      <Suspense fallback={null}>
        <Phase0 />
      </Suspense>
    );
  }
  return (
    <DataProvider>
      <Screen />
    </DataProvider>
  );
}
