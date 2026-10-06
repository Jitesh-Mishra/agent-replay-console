import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/graphql": process.env.GATEWAY_URL ?? "http://localhost:4000" } },
  preview: { proxy: { "/graphql": process.env.GATEWAY_URL ?? "http://localhost:4000" } },
  test: { environment: "jsdom", setupFiles: ["./test/setup.ts"] },
} as any);
