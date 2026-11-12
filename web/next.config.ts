import type { NextConfig } from "next";

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
// functions that use it (they're loaded from disk, not imported).
const ROUTER_FILES = [
  "./intent-router/intent_int8.onnx",
  "./intent-router/router_config.json",
  "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
  "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
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
