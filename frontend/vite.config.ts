/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      // The example circuits live in the repository's `examples/` directory,
      // one level above this project. They are read directly rather than
      // duplicated under src/, because the copy that used to live there drifted
      // from the originals and missed every circuit added after it was made.
      allow: [".."],
    },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
