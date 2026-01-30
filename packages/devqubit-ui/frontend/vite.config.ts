import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

export default defineConfig(({ mode }) => {
  const isLib = mode === 'lib';

  return {
    plugins: [
      react(),
      isLib && dts({
        insertTypesEntry: true,
        include: ['src'],
        exclude: ['src/main.tsx', 'src/App.tsx'],
      }),
    ].filter(Boolean),
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
      },
    },
    build: isLib
      ? {
          lib: {
            entry: resolve(__dirname, 'src/index.ts'),
            name: 'DevqubitUI',
            formats: ['es', 'cjs'],
            fileName: (format) => `index.${format === 'es' ? 'js' : 'cjs'}`,
          },
          rollupOptions: {
            external: ['react', 'react-dom', 'react-router-dom'],
            output: {
              globals: {
                react: 'React',
                'react-dom': 'ReactDOM',
                'react-router-dom': 'ReactRouterDOM',
              },
            },
          },
          cssCodeSplit: false,
        }
      : {
          outDir: 'dist',
        },
    server: {
      proxy: {
        '/api': 'http://localhost:8000',
        '/runs': 'http://localhost:8000',
        '/projects': 'http://localhost:8000',
        '/groups': 'http://localhost:8000',
        '/diff': 'http://localhost:8000',
        '/search': 'http://localhost:8000',
      },
    },
  };
});
