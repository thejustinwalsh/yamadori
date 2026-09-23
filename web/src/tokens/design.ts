// GENERATED from design/DESIGN.md frontmatter by web/build/gen-tokens.mjs.
// Do not edit: change DESIGN.md and run `npm run tokens`. A vitest test
// fails if this file and the frontmatter disagree.
//
// Plain values, for code that cannot read CSS variables (three.js materials,
// canvas). Components use tokens.stylex.ts instead.

export const color = {
  surface: "#0f131c",
  surfaceDim: "#0f131c",
  surfaceBright: "#353943",
  surfaceContainerLowest: "#0a0e17",
  surfaceContainerLow: "#181c25",
  surfaceContainer: "#1c2029",
  surfaceContainerHigh: "#262a34",
  surfaceContainerHighest: "#31353f",
  onSurface: "#dfe2ef",
  onSurfaceVariant: "#b9cbb9",
  inverseSurface: "#dfe2ef",
  inverseOnSurface: "#2c303a",
  outline: "#849585",
  outlineVariant: "#3b4b3d",
  surfaceTint: "#00e479",
  primary: "#f1ffef",
  onPrimary: "#003919",
  primaryContainer: "#00ff88",
  onPrimaryContainer: "#007139",
  inversePrimary: "#006d37",
  secondary: "#ffb2ba",
  onSecondary: "#670020",
  secondaryContainer: "#d4004b",
  onSecondaryContainer: "#ffe6e8",
  tertiary: "#eefeff",
  onTertiary: "#00363a",
  tertiaryContainer: "#5cf2ff",
  onTertiaryContainer: "#006d74",
  error: "#ffb4ab",
  onError: "#690005",
  errorContainer: "#93000a",
  onErrorContainer: "#ffdad6",
  primaryFixed: "#60ff99",
  primaryFixedDim: "#00e479",
  onPrimaryFixed: "#00210c",
  onPrimaryFixedVariant: "#005228",
  secondaryFixed: "#ffd9dc",
  secondaryFixedDim: "#ffb2ba",
  onSecondaryFixed: "#400011",
  onSecondaryFixedVariant: "#910030",
  tertiaryFixed: "#7df4ff",
  tertiaryFixedDim: "#00dbe9",
  onTertiaryFixed: "#002022",
  onTertiaryFixedVariant: "#004f54",
  background: "#0f131c",
  onBackground: "#dfe2ef",
  surfaceVariant: "#31353f",
} as const;

export type ColorToken = keyof typeof color;

export const rounded = {
  sm: "0.125rem",
  base: "0.25rem",
  md: "0.375rem",
  lg: "0.5rem",
  xl: "0.75rem",
  full: "9999px",
} as const;

export const spacing = {
  gutter: "0.75rem",
  margin: "1rem",
  spaceXs: "0.25rem",
  spaceSm: "0.5rem",
  spaceMd: "0.75rem",
  spaceLg: "1.25rem",
  spaceXl: "2rem",
} as const;

export const typography = {
  headlineXl: {"family":"'Syne', 'Futura', 'Avenir Next', system-ui, sans-serif","size":"36px","weight":"800","line":"42px","tracking":"-0.04em"},
  headlineXlMobile: {"family":"'Syne', 'Futura', 'Avenir Next', system-ui, sans-serif","size":"28px","weight":"800","line":"34px","tracking":"-0.03em"},
  headlineLg: {"family":"'Syne', 'Futura', 'Avenir Next', system-ui, sans-serif","size":"26px","weight":"700","line":"32px","tracking":"-0.02em"},
  headlineSm: {"family":"'Syne', 'Futura', 'Avenir Next', system-ui, sans-serif","size":"18px","weight":"700","line":"24px","tracking":"0em"},
  titleMd: {"family":"'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace","size":"15px","weight":"600","line":"20px","tracking":"-0.01em"},
  bodyLg: {"family":"'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace","size":"14px","weight":"400","line":"22px","tracking":"-0.01em"},
  bodySm: {"family":"'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace","size":"12px","weight":"400","line":"18px","tracking":"0em"},
  labelMd: {"family":"'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace","size":"11px","weight":"500","line":"14px","tracking":"0.06em"},
  labelXs: {"family":"'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace","size":"9px","weight":"700","line":"12px","tracking":"0.12em"},
} as const;
