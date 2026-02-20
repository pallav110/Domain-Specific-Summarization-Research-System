'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import axios from 'axios'
import DocumentUploader from '@/components/DocumentUploader'
import DocumentList from '@/components/DocumentList'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  FileText,
  BarChart3,
  FlaskConical,
  ArrowRight,
} from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Stats {
  total_documents: number
  total_summaries: number
  total_experiments: number
}

export default function Home() {
  const [uploadedDocId, setUploadedDocId] = useState<number | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios
      .get<Stats>(`${API_URL}/api/v1/dashboard/stats`)
      .then((res) => setStats(res.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="flex h-[calc(100dvh-64px)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2">
        <h1 className="text-lg font-semibold">Home</h1>
        <div className="flex items-center gap-2">
          <Button variant="outline" asChild title="View all documents">
            <Link href="/documents">
              <FileText className="mr-1.5 h-3.5 w-3.5" />
              Documents
            </Link>
          </Button>
          <Button asChild title="Run a new experiment">
            <Link href="/experiments">
              <FlaskConical className="mr-1.5 h-3.5 w-3.5" />
              New Experiment
            </Link>
          </Button>
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl space-y-6 px-4 py-4">
          {/* Quick Links */}
          <div className="grid grid-cols-3 gap-2">
            <Link
              href="/documents"
              className="group flex items-center gap-3 rounded-md border bg-background px-3 py-2.5 transition-colors hover:bg-muted/50"
              title="Upload and manage documents"
            >
              <FileText className="h-4 w-4 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium">Documents</p>
                <p className="text-[11px] text-muted-foreground">Upload & classify</p>
              </div>
              <ArrowRight className="h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
            </Link>
            <Link
              href="/experiments"
              className="group flex items-center gap-3 rounded-md border bg-background px-3 py-2.5 transition-colors hover:bg-muted/50"
              title="Create and run experiments"
            >
              <FlaskConical className="h-4 w-4 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium">Experiments</p>
                <p className="text-[11px] text-muted-foreground">Run comparisons</p>
              </div>
              <ArrowRight className="h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
            </Link>
            <Link
              href="/dashboard"
              className="group flex items-center gap-3 rounded-md border bg-background px-3 py-2.5 transition-colors hover:bg-muted/50"
              title="View analytics and metrics"
            >
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium">Dashboard</p>
                <p className="text-[11px] text-muted-foreground">Metrics & trends</p>
              </div>
              <ArrowRight className="h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
            </Link>
          </div>

          {/* Upload + Research Focus */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div>
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-xs font-medium uppercase text-muted-foreground">Upload Document</h2>
                <span className="text-[11px] text-muted-foreground">PDF or TXT</span>
              </div>
              <DocumentUploader onUploadSuccess={(id) => setUploadedDocId(id)} />
            </div>
            <div>
              <h2 className="mb-2 text-xs font-medium uppercase text-muted-foreground">Research Focus</h2>
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex items-start gap-2">
                  <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <span>Compare BART, PEGASUS, Gemini, Legal-BERT+PEGASUS, Clinical-BERT+PEGASUS</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <span>Evaluate with ROUGE, BERTScore, factuality, semantic similarity</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <span>Gemini as LLM baseline for quality and coverage</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <span>Export CSV/JSON for statistical analysis</span>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Documents */}
          <div>
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-xs font-medium uppercase text-muted-foreground">Recent Documents</h2>
              <Link href="/documents" className="text-[11px] font-medium text-primary hover:underline">
                View all
              </Link>
            </div>
            <DocumentList limit={5} highlightId={uploadedDocId} />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t bg-muted/30 px-4 py-2">
        <div className="grid grid-cols-3 gap-4">
          <div title="Total documents uploaded to the system">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Documents</p>
            {loading ? <Skeleton className="mt-1 h-4 w-8" /> : <p className="text-sm font-semibold">{stats?.total_documents ?? 0}</p>}
          </div>
          <div title="Total summaries generated across all models">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Summaries</p>
            {loading ? <Skeleton className="mt-1 h-4 w-8" /> : <p className="text-sm font-semibold">{stats?.total_summaries ?? 0}</p>}
          </div>
          <div title="Total experiments run comparing models">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Experiments</p>
            {loading ? <Skeleton className="mt-1 h-4 w-8" /> : <p className="text-sm font-semibold">{stats?.total_experiments ?? 0}</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
