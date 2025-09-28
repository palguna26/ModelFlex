import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // FastAPI backend
        changeOrigin: true,
        secure: false,
        // ❌ remove rewrite if your backend route starts with /api
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
