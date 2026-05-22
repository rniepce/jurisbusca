import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer'

// https://vite.dev/config/
// Set ANALYZE=1 to emit dist/stats.html after a production build.
export default defineConfig({
    plugins: [
        react(),
        process.env.ANALYZE === '1' && visualizer({
            filename: 'dist/stats.html',
            template: 'treemap',
            gzipSize: true,
            brotliSize: true,
            open: false,
        }),
    ].filter(Boolean),
    server: {
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom', 'react-router-dom'],
                    'supabase': ['@supabase/supabase-js'],
                    'docx-vendor': ['docx', 'file-saver'],
                    'icons': ['react-icons'],
                },
            },
        },
        chunkSizeWarningLimit: 600,
    },
})
