'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { FileText, FlaskConical, BarChart3, Home, TrendingUp, Table2 } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/documents', label: 'Documents', icon: FileText },
  { href: '/experiments', label: 'Experiments', icon: FlaskConical },
  { href: '/results', label: 'Results', icon: Table2 },
  { href: '/statistics', label: 'Statistics', icon: TrendingUp },
  { href: '/dashboard', label: 'Dashboard', icon: BarChart3 },
]

export default function Navigation() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 flex h-16 shrink-0 items-center border-b bg-background px-4">
      <Link href="/" className="mr-8 flex items-center gap-2" title="Home">
        <div className="rounded-md bg-primary p-1.5">
          <FileText className="h-4 w-4 text-primary-foreground" />
        </div>
        <span className="text-sm font-semibold">SumResearch</span>
      </Link>

      <nav className="flex items-center gap-1">
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href || (href !== '/' && pathname.startsWith(href))
          return (
            <Link
              key={href}
              href={href}
              title={label}
              className={cn(
                'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{label}</span>
            </Link>
          )
        })}
      </nav>
    </header>
  )
}
