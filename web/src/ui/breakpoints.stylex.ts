// Breakpoints as StyleX constants. They live in a .stylex file because StyleX
// statically evaluates media-query keys at build time, and it can only follow
// an import into a .stylex module (importing them from Bento.tsx failed the
// build with "Could not resolve the path to the imported file").
import * as stylex from '@stylexjs/stylex';

export const MQ = stylex.defineConsts({
  tablet: '@media (min-width: 768px) and (max-width: 1199px)',
  desktop: '@media (min-width: 1200px)',
});
