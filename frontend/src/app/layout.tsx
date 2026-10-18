import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Codebase RAG",
  description:
    "Ask questions about your code: code-aware chunks in ChromaDB, hybrid search, reranking and cited answers by Claude, measured by an evaluation suite.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
