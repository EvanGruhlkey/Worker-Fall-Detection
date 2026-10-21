import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Send /api requests (data, live stream, snapshots) to the FastAPI backend,
  // so the app can use relative URLs exactly as the backend returns them.
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
