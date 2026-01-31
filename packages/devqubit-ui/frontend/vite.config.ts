import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { cpSync, rmSync, existsSync } from 'fs';
import dts from 'vite-plugin-dts';

// Plugin to copy build output to Python static folder
function copyToStatic() {
  return {
    name: 'copy-to-static',
    closeBundle() {
      const staticDir = resolve(__dirname, '../src/devqubit_ui/static');
      const distDir = resolve(__dirname, 'dist');

      // Only copy for app build (not lib build)
      if (!existsSync(resolve(distDir, 'index.html'))) {
        return;
      }

      console.log('\n📦 Copying build to static folder...');

      // Clear existing static files (except .gitkeep)
      if (existsSync(staticDir)) {
        const files = ['index.html', 'assets'];
        files.forEach(file => {
          const path = resolve(staticDir, file);
          if (existsSync(path)) {
            rmSync(path, { recursive: true, force: true });
          }
        });
      }

      // Copy new build
      cpSync(resolve(distDir, 'index.html'), resolve(staticDir, 'index.html'));
      cpSync(resolve(distDir, 'assets'), resolve(staticDir, 'assets'), { recursive: true });

      console.log('✅ Static files updated!\n');
    }
  };
}

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
      !isLib && copyToStatic(),
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
              assetFileNames: 'style.[ext]',
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
      },
    },
  };
});
