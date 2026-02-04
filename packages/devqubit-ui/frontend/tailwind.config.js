/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  safelist: [
    // Button variants - these are generated dynamically
    'btn-primary',
    'btn-secondary',
    'btn-danger',
    'btn-ghost',
    'btn-ghost-danger',
    // Badge variants
    'badge-success',
    'badge-danger',
    'badge-warning',
    'badge-info',
    'badge-gray',
    'badge-neutral',
    // Alert variants
    'alert-success',
    'alert-danger',
    'alert-warning',
    'alert-info',
    // Diff page result banner
    'bg-success/10',
    'bg-warning/10',
    'border-success/20',
    'border-warning/20',
    'text-success',
    'text-warning',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2E9D6B',
          dark: '#257D56',
          light: '#3DB87E',
          bg: '#F7FAF9',
        },
        danger: {
          DEFAULT: '#DC4A4A',
          dark: '#C43C3C',
          bg: '#FEF2F2',
        },
        warning: {
          DEFAULT: '#D97706',
          bg: '#FFFBEB',
        },
        info: {
          DEFAULT: '#2563EB',
          bg: '#EFF6FF',
        },
        success: {
          DEFAULT: '#059669',
          bg: '#ECFDF5',
        },
        gray: {
          50: '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#111827',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
        mono: ['SF Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      fontSize: {
        xs: '0.75rem',
        sm: '0.8125rem',
        base: '0.875rem',
        lg: '1rem',
        xl: '1.125rem',
        '2xl': '1.5rem',
      },
      maxWidth: {
        container: '1400px',
      },
    },
  },
  plugins: [],
};
