import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Inter } from "next/font/google";
import "./globals.css";

const bricolage = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-heading",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Kelsie — your ambient AI sidekick",
  description:
    "Kelsie remembers what you're working on and makes sure you never miss a deadline.",
};

export const viewport: Viewport = {
  themeColor: "#F7F6F2",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${bricolage.variable} ${inter.variable}`}
      // Keeps smooth scrolling for in-page links but not page changes.
      data-scroll-behavior="smooth"
    >
      <body>{children}</body>
    </html>
  );
}
