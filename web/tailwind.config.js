/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'term-bg': '#0b0d10',
        'term-panel': '#13171c',
        'term-border': '#1f2630',
        'term-fg': '#d8dde3',
        'term-muted': '#7d8693',
        'term-amber': '#ffb000',
        'term-cyan': '#27e0c5',
        'term-magenta': '#ff5fa2',
        'term-green': '#3ddc84',
        'term-red': '#ff5757',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'IBM Plex Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};
