import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navigation from '@/components/Navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Domain-Specific Summarization Research System',
  description: 'Research system for legal and medical document summarization',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-slate-50">
          <div className="flex min-h-screen">
            <Navigation />
            <main className="flex-1 px-4 py-6 md:px-8 md:py-8">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  )
}
