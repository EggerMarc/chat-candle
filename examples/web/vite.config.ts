import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The model runs in a Web Worker; the wasm is imported as a URL asset.
export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
  optimizeDeps: {
    // The wasm-bindgen glue is a generated local module, not a pre-bundle dep.
    exclude: ["./src/pkg/chat_candle.js"],
  },
});
