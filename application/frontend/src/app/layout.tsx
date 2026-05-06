import type { Metadata } from "next";
import { Inter } from "next/font/google";

import { Providers } from "./providers";
import "@/styles/globals.css";

const inter = Inter({
  subsets: ["latin", "latin-ext"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "MedVoicePL — Pulpit lekarza",
  description:
    "Asystent kliniczny: nagrywanie wizyty, automatyczna notatka SOAP, sugestie leków z refundacją NFZ.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pl" className={inter.variable}>
      <body className="min-h-screen font-sans antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
