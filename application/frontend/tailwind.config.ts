import type { Config } from "tailwindcss";

/**
 * MedVoicePL design tokens — minimalist medical aesthetic.
 *
 * Palette: monochrome slate + calm sky-600 accent. No gradients, no
 * purples. NFZ refund tiers get semantic semaphore colors (emerald,
 * sky, amber, slate-400).
 *
 * Typography: Inter via `next/font/google`. Default Tailwind scale.
 *
 * Spacing: generous (cards p-6+, gap-6 between sections, max-w-5xl).
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Page surfaces
        "page-bg": "#f8fafc", // slate-50 — main app background
        "card-bg": "#ffffff",
        // Accent — calm clinical blue
        accent: {
          DEFAULT: "#0284c7", // sky-600
          hover: "#0369a1",
          fg: "#ffffff",
          subtle: "#e0f2fe", // sky-100
        },
        // Refund-tier semantics
        refund: {
          full: "#059669", // emerald-600  — 100%
          partial: "#0284c7", // sky-600   — 70% / 50%
          flat: "#d97706", // amber-600    — ryczałt
          none: "#64748b", // slate-500    — nierefundowane
        },
        // Status semantics
        ok: "#059669",
        warn: "#d97706",
        danger: "#dc2626",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 1px rgba(15, 23, 42, 0.03)",
        "card-hover":
          "0 4px 12px rgba(15, 23, 42, 0.08), 0 2px 4px rgba(15, 23, 42, 0.04)",
      },
    },
  },
  plugins: [],
};

export default config;
