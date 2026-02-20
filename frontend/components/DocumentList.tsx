'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import Link from 'next/link'
import { FileText } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Document {
  id: number
  filename: string
  original_filename: string
  file_size: number
  file_type: string
  word_count: number
  detected_domain: string
  domain_confidence: number
  upload_timestamp: string
  processed: number
}

interface Props {
  limit?: number
  highlightId?: number | null
}

const domainDot: Record<string, string> = {
  legal: 'bg-sky-500',
  medical: 'bg-emerald-500',
}

export default function DocumentList({ limit, highlightId }: Props) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const params = limit ? { limit } : {}
    axios
      .get<Document[]>(`${API_URL}/api/v1/documents`, { params })
      .then((res) => setDocuments(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load'))
      .finally(() => setLoading(false))
  }, [limit])

  if (loading) {
    return (
      <div className="space-y-1">
        {Array.from({ length: limit || 3 }).map((_, i) => (
          <div key={i} className="flex items-center gap-3 px-2 py-2">
            <Skeleton className="h-3.5 w-3.5 rounded" />
            <Skeleton className="h-3 w-40" />
            <Skeleton className="ml-auto h-3 w-16" />
          </div>
        ))}
      </div>
    )
  }

  if (error) {
    return <p className="py-4 text-center text-xs text-destructive">{error}</p>
  }

  if (documents.length === 0) {
    return (
      <div className="py-8 text-center">
        <FileText className="mx-auto h-8 w-8 text-muted-foreground/50" />
        <p className="mt-2 text-xs text-muted-foreground">No documents uploaded yet</p>
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded-md border">
      <table className="w-full">
        <thead>
          <tr className="border-b bg-muted/30">
            <th className="px-3 py-1.5 text-left text-[11px] font-medium uppercase text-muted-foreground">Name</th>
            <th className="px-3 py-1.5 text-left text-[11px] font-medium uppercase text-muted-foreground" title="Detected document domain">Domain</th>
            <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Word count of original document">Words</th>
            <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground">Date</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr
              key={doc.id}
              className={cn(
                'cursor-pointer transition-colors hover:bg-muted/50',
                highlightId === doc.id && 'bg-primary/5'
              )}
            >
              <td className="px-3 py-1.5">
                <Link href={`/documents/${doc.id}`} className="block text-xs font-medium truncate max-w-[200px]">
                  {doc.original_filename}
                </Link>
              </td>
              <td className="px-3 py-1.5">
                <div className="flex items-center gap-1.5">
                  <div className={cn('h-1.5 w-1.5 rounded-full', domainDot[doc.detected_domain?.toLowerCase()] || 'bg-muted-foreground')} />
                  <span className="text-xs text-muted-foreground">{doc.detected_domain || 'unknown'}</span>
                </div>
              </td>
              <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">{doc.word_count.toLocaleString()}</td>
              <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">{new Date(doc.upload_timestamp).toLocaleDateString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
