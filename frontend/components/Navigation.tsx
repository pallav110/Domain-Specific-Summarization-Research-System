'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import {
  FileText,
  FlaskConical,
  BarChart3,
  Home,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'

export default function Navigation() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)

  const navItems = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/documents', label: 'Documents', icon: FileText },
    { href: '/experiments', label: 'Experiments', icon: FlaskConical },
    { href: '/dashboard', label: 'Dashboard', icon: BarChart3 },
  ]

  return (
    <aside
      className={`sticky top-0 hidden h-screen shrink-0 border-r border-slate-200 bg-white shadow-sm md:flex md:flex-col ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-4">
        <Link href="/" className="flex items-center gap-3">
          <div className="rounded-lg bg-blue-700 p-2">
            <FileText className="h-5 w-5 text-white" />
          </div>
          {!collapsed && (
            <span className="text-sm font-semibold text-slate-900">
              Research Summarization
            </span>
          )}
        </Link>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="rounded-md border border-slate-200 p-1 text-slate-500 hover:text-slate-700"
          aria-label="Toggle sidebar"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      <nav className="flex-1 px-3 py-4">
        <div className="space-y-1">
          {navItems.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href
            return (
              <Link
                key={href}
                href={href}
                className={`group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <Icon className="h-5 w-5" />
                {!collapsed && <span>{label}</span>}
              </Link>
            )
          })}
        </div>
      </nav>

    </aside>
  )
}
