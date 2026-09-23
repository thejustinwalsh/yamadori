// Typography roles from DESIGN.md frontmatter, as StyleX styles. Components
// compose these; nobody writes a font-size by hand.
import * as stylex from '@stylexjs/stylex';
import { colors, type } from '../tokens/tokens.stylex';

export const text = stylex.create({
  headlineXl: {
    fontFamily: type.headlineXlFamily,
    fontSize: type.headlineXlSize,
    fontWeight: type.headlineXlWeight,
    lineHeight: type.headlineXlLine,
    letterSpacing: type.headlineXlTracking,
  },
  headlineXlMobile: {
    fontFamily: type.headlineXlMobileFamily,
    fontSize: type.headlineXlMobileSize,
    fontWeight: type.headlineXlMobileWeight,
    lineHeight: type.headlineXlMobileLine,
    letterSpacing: type.headlineXlMobileTracking,
  },
  headlineLg: {
    fontFamily: type.headlineLgFamily,
    fontSize: type.headlineLgSize,
    fontWeight: type.headlineLgWeight,
    lineHeight: type.headlineLgLine,
    letterSpacing: type.headlineLgTracking,
  },
  headlineSm: {
    fontFamily: type.headlineSmFamily,
    fontSize: type.headlineSmSize,
    fontWeight: type.headlineSmWeight,
    lineHeight: type.headlineSmLine,
    letterSpacing: type.headlineSmTracking,
  },
  titleMd: {
    fontFamily: type.titleMdFamily,
    fontSize: type.titleMdSize,
    fontWeight: type.titleMdWeight,
    lineHeight: type.titleMdLine,
    letterSpacing: type.titleMdTracking,
  },
  bodyLg: {
    fontFamily: type.bodyLgFamily,
    fontSize: type.bodyLgSize,
    fontWeight: type.bodyLgWeight,
    lineHeight: type.bodyLgLine,
    letterSpacing: type.bodyLgTracking,
  },
  bodySm: {
    fontFamily: type.bodySmFamily,
    fontSize: type.bodySmSize,
    fontWeight: type.bodySmWeight,
    lineHeight: type.bodySmLine,
    letterSpacing: type.bodySmTracking,
  },
  labelMd: {
    fontFamily: type.labelMdFamily,
    fontSize: type.labelMdSize,
    fontWeight: type.labelMdWeight,
    lineHeight: type.labelMdLine,
    letterSpacing: type.labelMdTracking,
    textTransform: 'uppercase',
  },
  labelXs: {
    fontFamily: type.labelXsFamily,
    fontSize: type.labelXsSize,
    fontWeight: type.labelXsWeight,
    lineHeight: type.labelXsLine,
    letterSpacing: type.labelXsTracking,
    textTransform: 'uppercase',
  },
  kanji: {
    fontFamily: "'Noto Serif JP', 'Yu Mincho', 'Hiragino Mincho ProN', serif",
    fontWeight: 700,
  },
  num: { fontVariantNumeric: 'tabular-nums' },
  // colour roles
  onSurface: { color: colors.onSurface },
  muted: { color: colors.onSurfaceVariant },
  outline: { color: colors.outline },
  primary: { color: colors.primary },
  moss: { color: colors.primaryContainer },
  cyan: { color: colors.tertiaryContainer },
  crimson: { color: colors.secondaryContainer },
  rose: { color: colors.secondary },
  error: { color: colors.error },
});
