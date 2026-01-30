/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#39B376',
          dark: '#2D9A64',
          light: '#4FD194',
          bg: '#EDFAF3',
        },
        danger: {
          DEFAULT: '#ef4444',
          dark: '#dc2626',
          bg: '#fee2e2',
        },
        warning: {
          DEFAULT: '#f59e0b',
          bg: '#fef3c7',
        },
        info: {
          DEFAULT: '#3b82f6',
          bg: '#dbeafe',
        },
        success: {
          DEFAULT: '#10b981',
          bg: '#d1fae5',
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
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
        mono: ['SF Mono', 'Monaco', 'Courier New', 'monospace'],
      },
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
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
