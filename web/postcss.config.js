import { stylexExtractBabelConfig, WEB_ROOT } from './build/stylex-options.js';

export default {
  plugins: {
    '@stylexjs/postcss-plugin': {
      cwd: WEB_ROOT,
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/**/*.test.{ts,tsx}'],
      babelConfig: stylexExtractBabelConfig,
    },
  },
};
