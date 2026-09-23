import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata = {
  title: "D.A.R.P.A.N — Digital Advanced Recommendation for Procurement and Allied Norms",
  description:
    "D.A.R.P.A.N: Digital Advanced Recommendation for Procurement and Allied Norms. Identify applicable Indian Standards, allied references, mandatory certifications, testing laboratories, and procurement requirements.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#0055A4",
};

import { ThemeProvider } from "./context/ThemeContext";

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} h-full`}>
      <body className="min-h-full flex flex-col font-(family-name:--font-inter) antialiased transition-colors duration-200">
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
