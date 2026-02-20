'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { ArrowUpDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface ModelMetricStats {
  mean: number
  std: number
  min: number
  max: number
  count: number
}

interface ModelStats {
  [metric: string]: ModelMetricStats
}

interface Ranking {
  model: string
  mean_score: number
}

interface StatsResponse {
  model_statistics: Record<string, ModelStats>
  domain_analysis: Record<string, Record<string, ModelStats>>
  rankings: Record<string, Ranking[]>
}

const METRIC_LABELS: Record<string, string> = {
  rouge_1_f: 'ROUGE-1 F1',
  rouge_2_f: 'ROUGE-2 F1',
  rouge_l_f: 'ROUGE-L F1',
  bertscore_f1: 'BERTScore F1',
  factuality_score: 'Factuality',
  semantic_similarity: 'Semantic Sim.',
  compression_ratio: 'Compression',
  generation_time: 'Gen Time (s)',
}

const METRIC_TOOLTIPS: Record<string, string> = {
  rouge_1_f: 'Unigram overlap F1 between summary and source',
  rouge_2_f: 'Bigram overlap F1 between summary and source',
  rouge_l_f: 'Longest common subsequence F1 between summary and source',
  bertscore_f1: 'Semantic similarity using contextual embeddings (BERT)',
  factuality_score: 'Factual consistency of summary with source document',
  semantic_similarity: 'Contextual semantic alignment score',
  compression_ratio: 'Ratio of summary length to source length',
  generation_time: 'Time in seconds to generate the summary',
}

type TabView = 'overview' | 'domain' | 'rankings'

export default function StatisticsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [domain, setDomain] = useState<string>('')
  const [tab, setTab] = useState<TabView>('overview')
  const [sortMetric, setSortMetric] = useState('rouge_l_f')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    const params = domain ? { domain } : {}
    axios
      .get<StatsResponse>(`${API_URL}/api/v1/statistics/analysis`, { params })
      .then((res) => setStats(res.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [domain])

  const tabs: { value: TabView; label: string }[] = [
    { value: 'overview', label: 'Model Statistics' },
    { value: 'domain', label: 'Domain Analysis' },
    { value: 'rankings', label: 'Rankings' },
  ]

  const formatVal = (v: number | undefined, metric: string) => {
    if (v === undefined || v === null) return '—'
    if (metric === 'generation_time') return v.toFixed(2)
    if (metric === 'compression_ratio') return v.toFixed(3)
    return v.toFixed(4)
  }

  const handleSort = (metric: string) => {
    if (sortMetric === metric) setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    else { setSortMetric(metric); setSortDir('desc') }
  }

  const getSortedModels = () => {
    if (!stats?.model_statistics) return []
    return Object.entries(stats.model_statistics).sort(([, a], [, b]) => {
      const aVal = a[sortMetric]?.mean ?? 0
      const bVal = b[sortMetric]?.mean ?? 0
      return sortDir === 'desc' ? bVal - aVal : aVal - bVal
    })
  }

  if (loading) {
    return (
      <div className="flex h-[calc(100dvh-64px)] flex-col">
        <div className="flex items-center gap-2 border-b bg-muted/30 px-4 py-2">
          <Skeleton className="h-5 w-48" />
        </div>
        <div className="flex-1 px-4 py-4">
          <div className="space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="flex h-[calc(100dvh-64px)] items-center justify-center">
        <div className="text-center">
          <p className="text-sm text-muted-foreground">No statistical data available yet.</p>
          <p className="mt-1 text-xs text-muted-foreground">Run some experiments first to generate data.</p>
        </div>
      </div>
    )
  }

  const models = getSortedModels()
  const metrics = Object.keys(METRIC_LABELS)

  return (
    <div className="flex h-[calc(100dvh-64px)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2">
        <div className="flex items-center gap-2">
          <h1 className="text-lg font-semibold">Statistical Analysis</h1>
          <span className="text-xs text-muted-foreground">
            {Object.keys(stats.model_statistics || {}).length} models
          </span>
        </div>
        <select
          value={domain}
          onChange={(e) => { setDomain(e.target.value); setLoading(true) }}
          className="h-8 rounded-md border bg-background px-2 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          title="Filter statistics by document domain"
        >
          <option value="">All Domains</option>
          <option value="legal">Legal</option>
          <option value="medical">Medical</option>
        </select>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b px-4 py-1.5">
        {tabs.map((t) => (
          <button
            key={t.value}
            onClick={() => setTab(t.value)}
            className={cn(
              'rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
              tab === t.value ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent'
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {tab === 'overview' && (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-muted/30">
                  <th className="sticky left-0 bg-muted/30 px-4 py-1.5 text-left text-[11px] font-medium uppercase text-muted-foreground">Model</th>
                  {metrics.map((m) => (
                    <th
                      key={m}
                      className="cursor-pointer px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground hover:text-foreground"
                      title={METRIC_TOOLTIPS[m]}
                      onClick={() => handleSort(m)}
                    >
                      <div className="flex items-center justify-end gap-1">
                        {METRIC_LABELS[m]}
                        {sortMetric === m && <ArrowUpDown className="h-2.5 w-2.5" />}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map(([model, modelStats]) => (
                  <tr key={model} className="border-b border-transparent transition-colors hover:bg-muted/50">
                    <td className="sticky left-0 bg-background px-4 py-2 text-xs font-semibold">{model.toUpperCase()}</td>
                    {metrics.map((m) => {
                      const s = modelStats[m]
                      return (
                        <td
                          key={m}
                          className="px-3 py-2 text-right"
                          title={s ? `Mean: ${formatVal(s.mean, m)} | Std: ${formatVal(s.std, m)} | Min: ${formatVal(s.min, m)} | Max: ${formatVal(s.max, m)} | n=${s.count}` : 'No data'}
                        >
                          <span className="text-xs font-medium">{formatVal(s?.mean, m)}</span>
                          {s?.std !== undefined && s.count > 1 && (
                            <span className="ml-1 text-[10px] text-muted-foreground">&plusmn;{formatVal(s.std, m)}</span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="px-4 py-2 text-[10px] text-muted-foreground">
              Hover over cells for full statistics (mean, std dev, min, max, count). Click column headers to sort.
            </p>
          </div>
        )}

        {tab === 'domain' && (
          <div className="space-y-6 px-4 py-4">
            {stats.domain_analysis && Object.keys(stats.domain_analysis).length > 0 ? (
              Object.entries(stats.domain_analysis).map(([domainName, domainModels]) => (
                <div key={domainName}>
                  <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase text-muted-foreground">
                    <div className={cn('h-1.5 w-1.5 rounded-full', domainName === 'legal' ? 'bg-sky-500' : domainName === 'medical' ? 'bg-emerald-500' : 'bg-muted-foreground')} />
                    {domainName} Domain
                  </h3>
                  <div className="overflow-x-auto rounded-md border">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b bg-muted/30">
                          <th className="px-3 py-1.5 text-left text-[11px] font-medium uppercase text-muted-foreground">Model</th>
                          <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="ROUGE-L F1 score">ROUGE-L</th>
                          <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="BERTScore F1">BERTScore</th>
                          <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Factuality score">Factuality</th>
                          <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Semantic similarity">Sem. Sim.</th>
                          <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Average generation time">Time (s)</th>
                          <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Number of evaluations">n</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(domainModels).map(([model, ms]) => (
                          <tr key={model} className="border-b border-transparent hover:bg-muted/50">
                            <td className="px-3 py-1.5 text-xs font-semibold">{model.toUpperCase()}</td>
                            <td className="px-3 py-1.5 text-right text-xs">{formatVal(ms.rouge_l_f?.mean, 'rouge_l_f')}</td>
                            <td className="px-3 py-1.5 text-right text-xs">{formatVal(ms.bertscore_f1?.mean, 'bertscore_f1')}</td>
                            <td className="px-3 py-1.5 text-right text-xs">{formatVal(ms.factuality_score?.mean, 'factuality_score')}</td>
                            <td className="px-3 py-1.5 text-right text-xs">{formatVal(ms.semantic_similarity?.mean, 'semantic_similarity')}</td>
                            <td className="px-3 py-1.5 text-right text-xs">{formatVal(ms.generation_time?.mean, 'generation_time')}</td>
                            <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">{ms.rouge_l_f?.count ?? 0}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))
            ) : (
              <div className="py-8 text-center">
                <p className="text-xs text-muted-foreground">No domain-specific analysis available yet.</p>
              </div>
            )}
          </div>
        )}

        {tab === 'rankings' && (
          <div className="grid grid-cols-1 gap-4 px-4 py-4 md:grid-cols-2">
            {stats.rankings && Object.entries(stats.rankings).map(([metric, ranked]) => (
              <div key={metric} className="rounded-md border">
                <div className="border-b bg-muted/30 px-3 py-1.5">
                  <h3 className="text-[11px] font-medium uppercase text-muted-foreground"
                    title={METRIC_TOOLTIPS[metric] || metric}>
                    {METRIC_LABELS[metric] || metric}
                  </h3>
                </div>
                <div>
                  {ranked.map((r, i) => (
                    <div
                      key={r.model}
                      className="flex items-center justify-between border-b border-transparent px-3 py-1.5 hover:bg-muted/50"
                    >
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          'flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-bold',
                          i === 0 ? 'bg-amber-100 text-amber-700' : 'bg-muted text-muted-foreground'
                        )}>
                          {i + 1}
                        </span>
                        <span className="text-xs font-medium">{r.model.toUpperCase()}</span>
                      </div>
                      <span className="text-xs text-muted-foreground">{r.mean_score.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t bg-muted/30 px-4 py-2">
        <div className="grid grid-cols-4 gap-4">
          <div title="Number of models analyzed">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Models</p>
            <p className="text-sm font-semibold">{Object.keys(stats.model_statistics || {}).length}</p>
          </div>
          <div title="Number of ranking categories">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Metrics Ranked</p>
            <p className="text-sm font-semibold">{Object.keys(stats.rankings || {}).length}</p>
          </div>
          <div title="Number of domains analyzed">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Domains</p>
            <p className="text-sm font-semibold">{Object.keys(stats.domain_analysis || {}).length}</p>
          </div>
          <div title="Current domain filter applied">
            <p className="text-[10px] font-medium uppercase text-muted-foreground">Filter</p>
            <p className="text-sm font-semibold">{domain || 'All'}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
