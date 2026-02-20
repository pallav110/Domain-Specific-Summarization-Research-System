'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Download, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface DashboardStats {
  total_documents: number
  total_summaries: number
  total_experiments: number
  domains_distribution: Record<string, number>
  model_usage: Record<string, number>
  average_metrics: Record<string, number>
}

const COLORS = ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [exportDomain, setExportDomain] = useState('')

  useEffect(() => {
    axios
      .get<DashboardStats>(`${API_URL}/api/v1/dashboard/stats`)
      .then((res) => setStats(res.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex h-[calc(100dvh-64px)] flex-col">
        <div className="flex items-center gap-2 border-b bg-muted/30 px-4 py-2">
          <Skeleton className="h-5 w-40" />
        </div>
        <div className="flex-1 space-y-4 px-4 py-4">
          <div className="grid grid-cols-3 gap-2">
            {[1, 2, 3].map((i) => <Skeleton key={i} className="h-14 w-full" />)}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-48 w-full" />
            <Skeleton className="h-48 w-full" />
          </div>
          <Skeleton className="h-48 w-full" />
        </div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="flex h-[calc(100dvh-64px)] items-center justify-center">
        <p className="text-sm text-destructive">Failed to load dashboard data</p>
      </div>
    )
  }

  const domainData = Object.entries(stats.domains_distribution).map(([name, value]) => ({
    name: name.toUpperCase(),
    value,
  }))

  const modelData = Object.entries(stats.model_usage).map(([name, value]) => ({
    name: name.toUpperCase(),
    count: value,
  }))

  const metricsData = [
    { metric: 'ROUGE-1', score: stats.average_metrics.rouge_1 || 0 },
    { metric: 'ROUGE-2', score: stats.average_metrics.rouge_2 || 0 },
    { metric: 'ROUGE-L', score: stats.average_metrics.rouge_l || 0 },
    { metric: 'BERTScore', score: stats.average_metrics.bertscore || 0 },
    { metric: 'Factuality', score: stats.average_metrics.factuality || 0 },
  ]

  return (
    <div className="flex h-[calc(100dvh-64px)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2">
        <h1 className="text-lg font-semibold">Dashboard</h1>
        <div className="flex items-center gap-2">
          <select
            value={exportDomain}
            onChange={(e) => setExportDomain(e.target.value)}
            className="h-8 rounded-md border bg-background px-2 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
            title="Filter exports by domain"
          >
            <option value="">All Domains</option>
            <option value="legal">Legal</option>
            <option value="medical">Medical</option>
          </select>
          <Button variant="outline" asChild title="Download evaluation results as CSV">
            <a href={`${API_URL}/api/v1/export/csv${exportDomain ? `?domain=${exportDomain}` : ''}`} download>
              <Download className="mr-1.5 h-3.5 w-3.5" />
              CSV
            </a>
          </Button>
          <Button variant="outline" asChild title="Download evaluation results as JSON">
            <a href={`${API_URL}/api/v1/export/json${exportDomain ? `?domain=${exportDomain}` : ''}`} download>
              <Download className="mr-1.5 h-3.5 w-3.5" />
              JSON
            </a>
          </Button>
          <Button variant="outline" asChild title="View statistical analysis report">
            <a href={`${API_URL}/api/v1/statistics/analysis`} target="_blank" rel="noopener noreferrer">
              <TrendingUp className="mr-1.5 h-3.5 w-3.5" />
              Analysis
            </a>
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="mx-auto max-w-6xl space-y-6">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded-md border px-3 py-2" title="Total number of documents uploaded to the system">
              <p className="text-[10px] font-medium uppercase text-muted-foreground">Documents</p>
              <p className="text-xl font-semibold">{stats.total_documents}</p>
            </div>
            <div className="rounded-md border px-3 py-2" title="Total number of summaries generated across all models">
              <p className="text-[10px] font-medium uppercase text-muted-foreground">Summaries</p>
              <p className="text-xl font-semibold">{stats.total_summaries}</p>
            </div>
            <div className="rounded-md border px-3 py-2" title="Total number of experiments run">
              <p className="text-[10px] font-medium uppercase text-muted-foreground">Experiments</p>
              <p className="text-xl font-semibold">{stats.total_experiments}</p>
            </div>
          </div>

          {/* Charts Row 1 */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-md border px-3 py-2">
              <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="Distribution of documents across legal, medical, and other domains">Domain Distribution</h3>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={domainData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    dataKey="value"
                    style={{ fontSize: 10 }}
                  >
                    {domainData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-md border px-3 py-2">
              <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="Number of summaries generated per model">Model Usage</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={modelData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Bar dataKey="count" fill="#0ea5e9" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Average Metrics */}
          <div className="rounded-md border px-3 py-2">
            <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="Average evaluation scores across all summaries">Average Evaluation Metrics</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={metricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} formatter={(value: number) => value.toFixed(4)} />
                <Bar dataKey="score" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Research Insights */}
          <div>
            <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground">Research Insights</h3>
            <div className="space-y-1.5 text-xs text-muted-foreground">
              <div className="flex items-start gap-2" title="ROUGE-L measures content overlap between summary and source using longest common subsequence">
                <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                <span><strong className="text-foreground">Avg ROUGE-L:</strong> {(stats.average_metrics.rouge_l || 0).toFixed(4)} — content overlap</span>
              </div>
              <div className="flex items-start gap-2" title="BERTScore measures semantic similarity using contextual embeddings from BERT">
                <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                <span><strong className="text-foreground">Avg BERTScore:</strong> {(stats.average_metrics.bertscore || 0).toFixed(4)} — semantic similarity</span>
              </div>
              <div className="flex items-start gap-2" title="Factuality score measures how factually consistent the summary is with the source document">
                <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                <span><strong className="text-foreground">Avg Factuality:</strong> {(stats.average_metrics.factuality || 0).toFixed(4)} — factual consistency</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t bg-muted/30 px-4 py-2">
        <div className="grid grid-cols-5 gap-4">
          <div title="Total documents in the system">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Docs</p>
            <p className="text-sm font-semibold">{stats.total_documents}</p>
          </div>
          <div title="Total summaries generated">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Summaries</p>
            <p className="text-sm font-semibold">{stats.total_summaries}</p>
          </div>
          <div title="Average ROUGE-L F1 score across all evaluated summaries">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Avg ROUGE-L</p>
            <p className="text-sm font-semibold">{(stats.average_metrics.rouge_l || 0).toFixed(4)}</p>
          </div>
          <div title="Average BERTScore F1 across all evaluated summaries">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Avg BERT</p>
            <p className="text-sm font-semibold">{(stats.average_metrics.bertscore || 0).toFixed(4)}</p>
          </div>
          <div title="Average factuality score across all evaluated summaries">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Avg Fact.</p>
            <p className="text-sm font-semibold">{(stats.average_metrics.factuality || 0).toFixed(4)}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
