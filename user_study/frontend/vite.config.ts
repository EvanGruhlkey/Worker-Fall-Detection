import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "tailwindcss";

export default defineConfig({
  plugins: [react()],
  css: { postcss: { plugins: [tailwindcss("./flappy/tailwind.config.mjs")] } },
  build: { rollupOptions: { input: ["index.html", "flappy/index.html"] } },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
