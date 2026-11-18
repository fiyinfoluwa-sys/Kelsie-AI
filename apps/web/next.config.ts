import path from "node:path";
import type { NextConfig } from "next";

// The monorepo root (apps/web is two levels down). Dependencies are installed
// there, so Next.js needs it to trace and bundle files outside apps/web.
const REPO_ROOT = path.join(__dirname, "../..");

const securityHeaders = [
  // No other site may put Kelsie's pages in a frame (stops clickjacking on
  // the sign-in and delete-account forms).
  { key: "Content-Security-Policy", value: "frame-ancestors 'none'" },
  { key: "X-Frame-Options", value: "DENY" },
  // Browsers must use the declared file type instead of guessing.
  { key: "X-Content-Type-Options", value: "nosniff" },
  // Other sites see only "kelsie-ai-beta.vercel.app", never full URLs.
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // Always use HTTPS (ignored on http://localhost).
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains" },
  // The site doesn't need these browser features.
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=()" },
];

// Files the intent router reads at runtime, which Vercel must ship with the
// functions that use it (they're loaded from disk, not imported). Paths are
// relative to apps/web; ONNX Runtime lives in the repo-root node_modules.
const ROUTER_FILES = [
  "./intent-router/intent_int8.onnx",
  "./intent-router/router_config.json",
  "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
  "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  outputFileTracingRoot: REPO_ROOT,
  turbopack: { root: REPO_ROOT },
  // ONNX Runtime loads its WebAssembly from node_modules; don't bundle it.
  serverExternalPackages: ["onnxruntime-web"],
  outputFileTracingIncludes: {
    "/api/kelsie/message": ROUTER_FILES,
    "/api/kelsie/router-health": ROUTER_FILES,
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
