import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

const backend = 'http://127.0.0.1:5000';
export default defineConfig({
  plugins: [sveltekit()],
  server: { port: 5173, proxy: { '/api': backend, '/static': backend, '/reports': backend } },
  test: { include: ['src/**/*.test.ts'], environment: 'node' },
});
