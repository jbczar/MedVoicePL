/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Gateway runs at GATEWAY_URL.
  //
  // IMPORTANT: must use `localhost` (not `127.0.0.1`) to match the
  // frontend's origin host. Browsers treat `localhost` and `127.0.0.1`
  // as DIFFERENT sites for cookie/SameSite purposes, so a session
  // cookie set on `127.0.0.1` won't be sent on cross-site fetch from
  // `localhost:3000` even with `credentials: 'include'`. Both hostnames
  // resolve to the same IP, so the gateway (which listens on
  // 127.0.0.1:8080) accepts the connection just fine via `localhost`.
  env: {
    NEXT_PUBLIC_GATEWAY_URL: process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8080",
  },
};

export default nextConfig;
