'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import axios from 'axios'
import Link from 'next/link'
import { Loader2, ArrowLeft, BarChart3, Trophy, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface EvalMetrics {
  rouge_1_f: number | null
  rouge_1_p: number | null
  rouge_1_r: number | null
  rouge_2_f: number | null
  rouge_l_f: number | null
  rouge_l_p: number | null
  rouge_l_r: number | null
  bertscore_f1: number | null
  bertscore_precision: number | null
  bertscore_recall: number | null
  factuality_score: number | null
  compression_ratio: number | null
  semantic_similarity: number | null
}

interface ModelComparison {
  model_type: string
  model_name: string
  summary_id: number
  summary_length: number
  generation_time: number
  summary_text: string
  rouge_1_f: number | null
  rouge_2_f: number | null
  rouge_l_f: number | null
  rouge_1_p: number | null
  rouge_1_r: number | null
  rouge_l_p: number | null
  rouge_l_r: number | null
  bertscore_f1: number | null
  bertscore_precision: number | null
  bertscore_recall: number | null
  factuality_score: number | null
  semantic_similarity: number | null
  compression_ratio: number | null
}

// Raw API response shape (metrics nested under `metrics`)
interface APIModelComparison {
  model_type: string
  model_name: string
  summary_id: number
  summary_length: number
  generation_time: number
  summary_text: string
  metrics: EvalMetrics
}

interface APIComparisonResponse {
  document_id: number
  document_name: string
  domain: string
  word_count: number
  models: APIModelComparison[]
  best_overall: string
  recommendations: string[]
}

interface ComparisonResponse {
  document_id: number
  document_name: string
  domain: string
  word_count: number
  models: ModelComparison[]
  best_model: string
  recommendations: string[]
}

function transformAPIResponse(api: APIComparisonResponse): ComparisonResponse {
  return {
    document_id: api.document_id,
    document_name: api.document_name,
    domain: api.domain,
    word_count: api.word_count,
    best_model: api.best_overall,
    recommendations: api.recommendations,
    models: api.models.map(m => ({
      model_type: m.model_type,
      model_name: m.model_name,
      summary_id: m.summary_id,
      summary_length: m.summary_length,
      generation_time: m.generation_time,
      summary_text: m.summary_text,
      rouge_1_f: m.metrics?.rouge_1_f ?? null,
      rouge_2_f: m.metrics?.rouge_2_f ?? null,
      rouge_l_f: m.metrics?.rouge_l_f ?? null,
      rouge_1_p: m.metrics?.rouge_1_p ?? null,
      rouge_1_r: m.metrics?.rouge_1_r ?? null,
      rouge_l_p: m.metrics?.rouge_l_p ?? null,
      rouge_l_r: m.metrics?.rouge_l_r ?? null,
      bertscore_f1: m.metrics?.bertscore_f1 ?? null,
      bertscore_precision: m.metrics?.bertscore_precision ?? null,
      bertscore_recall: m.metrics?.bertscore_recall ?? null,
      factuality_score: m.metrics?.factuality_score ?? null,
      semantic_similarity: m.metrics?.semantic_similarity ?? null,
      compression_ratio: m.metrics?.compression_ratio ?? null,
    })),
  }
}

// Fallback types for manual fetching
interface Summary { id: number; document_id: number; model_type: string; model_name: string; summary_text: string; summary_length: number; generation_time: number; created_at: string }
interface Document { id: number; original_filename: string; word_count: number; detected_domain: string; domain_confidence: number }
interface Evaluation { rouge_1_f: number | null; rouge_2_f: number | null; rouge_l_f: number | null; rouge_1_p?: number | null; rouge_1_r?: number | null; rouge_l_p?: number | null; rouge_l_r?: number | null; bertscore_f1: number | null; bertscore_precision?: number | null; bertscore_recall?: number | null; factuality_score: number | null; semantic_similarity: number | null; compression_ratio: number | null }

const MODEL_COLORS: Record<string, string> = {
  bart: '#f97316', pegasus: '#8b5cf6', gemini: '#3b82f6', gpt: '#10b981',
  t5: '#ef4444', legal_bert_pegasus: '#06b6d4', clinical_bert_pegasus: '#ec4899',
}

export default function ComparePage() {
  const params = useParams()
  const documentId = params.id as string

  const [comparison, setComparison] = useState<ComparisonResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [evaluatingAll, setEvaluatingAll] = useState(false)
  const [showPR, setShowPR] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Try the compare API first
        const res = await axios.get<APIComparisonResponse>(`${API_URL}/api/v1/compare/${documentId}`)
        setComparison(transformAPIResponse(res.data))
      } catch {
        // Fallback: manually assemble from individual endpoints
        try {
          const [docRes, sumRes] = await Promise.all([
            axios.get<Document>(`${API_URL}/api/v1/documents/${documentId}`),
            axios.get<Summary[]>(`${API_URL}/api/v1/documents/${documentId}/summaries`),
          ])
          const models: ModelComparison[] = await Promise.all(
            sumRes.data.map(async (s) => {
              let ev: Evaluation = {} as Evaluation
              try { const r = await axios.get(`${API_URL}/api/v1/evaluations/summary/${s.id}`); ev = r.data } catch {}
              return {
                model_type: s.model_type, model_name: s.model_name, summary_id: s.id,
                summary_length: s.summary_length, generation_time: s.generation_time, summary_text: s.summary_text,
                rouge_1_f: ev.rouge_1_f ?? null, rouge_2_f: ev.rouge_2_f ?? null, rouge_l_f: ev.rouge_l_f ?? null,
                rouge_1_p: ev.rouge_1_p ?? null, rouge_1_r: ev.rouge_1_r ?? null,
                rouge_l_p: ev.rouge_l_p ?? null, rouge_l_r: ev.rouge_l_r ?? null,
                bertscore_f1: ev.bertscore_f1 ?? null, bertscore_precision: ev.bertscore_precision ?? null, bertscore_recall: ev.bertscore_recall ?? null,
                factuality_score: ev.factuality_score ?? null, semantic_similarity: ev.semantic_similarity ?? null,
                compression_ratio: ev.compression_ratio ?? null,
              }
            })
          )
          setComparison({
            document_id: docRes.data.id, document_name: docRes.data.original_filename,
            domain: docRes.data.detected_domain, word_count: docRes.data.word_count,
            models, best_model: '', recommendations: [],
          })
        } catch {}
      }
      setLoading(false)
    }
    fetchData()
  }, [documentId])

  const evaluateAll = async () => {
    if (!comparison) return
    setEvaluatingAll(true)
    try {
      for (const m of comparison.models) {
        if (m.rouge_l_f !== null) continue
        try { await axios.post(`${API_URL}/api/v1/evaluate/${m.summary_id}`) } catch {}
      }
      // Refresh
      const res = await axios.get<APIComparisonResponse>(`${API_URL}/api/v1/compare/${documentId}`)
      setComparison(transformAPIResponse(res.data))
    } catch {}
    setEvaluatingAll(false)
  }

  const getModelColor = (model: string) => MODEL_COLORS[model?.toLowerCase()] || '#6b7280'
  const fmt = (v: number | null) => v !== null && v !== undefined ? v.toFixed(4) : '—'

  if (loading) {
    return (
      <div className="flex h-[calc(100dvh-64px)] flex-col">
        <div className="flex items-center gap-2 border-b bg-muted/30 px-4 py-2"><Skeleton className="h-5 w-48" /></div>
        <div className="flex-1 space-y-4 px-4 py-4">
          <div className="grid grid-cols-3 gap-2">{[1,2,3].map(i=><Skeleton key={i} className="h-16"/>)}</div>
          <div className="grid grid-cols-2 gap-4"><Skeleton className="h-48"/><Skeleton className="h-48"/></div>
        </div>
      </div>
    )
  }

  if (!comparison || comparison.models.length === 0) {
    return (
      <div className="flex h-[calc(100dvh-64px)] flex-col items-center justify-center">
        <p className="text-sm text-destructive">No summaries found for this document.</p>
        <Link href={`/documents/${documentId}`} className="mt-2 text-xs text-primary hover:underline">Generate summaries first</Link>
      </div>
    )
  }

  const { models } = comparison
  const hasEvals = models.some(m => m.rouge_l_f !== null)

  const lengthData = models.map(m => ({ model: m.model_type.toUpperCase(), words: m.summary_length, time: parseFloat(m.generation_time?.toFixed(2) || '0') }))

  const rougeData = hasEvals ? [
    { metric: 'ROUGE-1', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), m.rouge_1_f ?? 0])) },
    { metric: 'ROUGE-2', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), m.rouge_2_f ?? 0])) },
    { metric: 'ROUGE-L', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), m.rouge_l_f ?? 0])) },
  ] : []

  const radarData = hasEvals ? [
    { metric: 'ROUGE-1', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), (m.rouge_1_f ?? 0) * 100])) },
    { metric: 'ROUGE-2', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), (m.rouge_2_f ?? 0) * 100])) },
    { metric: 'ROUGE-L', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), (m.rouge_l_f ?? 0) * 100])) },
    { metric: 'BERTScore', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), (m.bertscore_f1 ?? 0) * 100])) },
    { metric: 'Factuality', ...Object.fromEntries(models.map(m => [m.model_type.toUpperCase(), (m.factuality_score ?? 0) * 100])) },
  ] : []

  const weightedScore = (m: ModelComparison) => (m.rouge_l_f ?? 0) * 0.3 + (m.bertscore_f1 ?? 0) * 0.3 + (m.semantic_similarity ?? 0) * 0.2 + (m.factuality_score ?? 0) * 0.2

  return (
    <div className="flex h-[calc(100dvh-64px)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" asChild title="Back to document">
            <Link href={`/documents/${documentId}`}><ArrowLeft className="h-3.5 w-3.5" /></Link>
          </Button>
          <h1 className="text-lg font-semibold">Model Comparison</h1>
          <span className="text-xs text-muted-foreground">{comparison.document_name} &middot; {models.length} models</span>
          {comparison.best_model && (
            <span className="rounded-md bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-700" title="Best overall model determined by the backend">
              Best: {comparison.best_model.toUpperCase()}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {hasEvals && (
            <button
              onClick={() => setShowPR(!showPR)}
              className={cn('rounded-md px-2.5 py-1 text-xs font-medium transition-colors', showPR ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent')}
              title="Toggle Precision/Recall columns"
            >
              {showPR ? 'Hide P/R' : 'Show P/R'}
            </button>
          )}
          {!hasEvals && (
            <Button onClick={evaluateAll} disabled={evaluatingAll} title="Run evaluation metrics for all models">
              {evaluatingAll ? <><Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />Evaluating...</> : <><BarChart3 className="mr-1.5 h-3.5 w-3.5" />Evaluate All</>}
            </Button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="mx-auto max-w-6xl space-y-6">
          {/* Recommendations */}
          {comparison.recommendations && comparison.recommendations.length > 0 && (
            <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2">
              <h3 className="text-[11px] font-medium uppercase text-amber-700">Domain Recommendations</h3>
              {comparison.recommendations.map((r, i) => (
                <p key={i} className="mt-0.5 text-xs text-amber-800">{r}</p>
              ))}
            </div>
          )}

          {/* Full Metrics Table */}
          {hasEvals && (
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="sticky left-0 bg-muted/30 px-3 py-1.5 text-left text-[11px] font-medium uppercase text-muted-foreground">Model</th>
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="ROUGE-1 F1">R-1</th>
                    {showPR && <th className="px-2 py-1.5 text-right text-[10px] font-medium uppercase text-muted-foreground" title="ROUGE-1 Precision">R1-P</th>}
                    {showPR && <th className="px-2 py-1.5 text-right text-[10px] font-medium uppercase text-muted-foreground" title="ROUGE-1 Recall">R1-R</th>}
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="ROUGE-2 F1">R-2</th>
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="ROUGE-L F1">R-L</th>
                    {showPR && <th className="px-2 py-1.5 text-right text-[10px] font-medium uppercase text-muted-foreground" title="ROUGE-L Precision">RL-P</th>}
                    {showPR && <th className="px-2 py-1.5 text-right text-[10px] font-medium uppercase text-muted-foreground" title="ROUGE-L Recall">RL-R</th>}
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="BERTScore F1">BERT</th>
                    {showPR && <th className="px-2 py-1.5 text-right text-[10px] font-medium uppercase text-muted-foreground" title="BERTScore Precision">B-P</th>}
                    {showPR && <th className="px-2 py-1.5 text-right text-[10px] font-medium uppercase text-muted-foreground" title="BERTScore Recall">B-R</th>}
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Factuality score">Fact.</th>
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Semantic similarity">Sem.</th>
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Words in summary">Words</th>
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Generation time (seconds)">Time</th>
                    <th className="px-3 py-1.5 text-right text-[11px] font-medium uppercase text-muted-foreground" title="Weighted score: 30% R-L + 30% BERT + 20% Sem + 20% Fact">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {[...models].sort((a, b) => weightedScore(b) - weightedScore(a)).map((m, i) => (
                    <tr key={m.summary_id ?? `${m.model_type}-${i}`} className={cn('border-b border-transparent hover:bg-muted/50', comparison.best_model === m.model_type && 'bg-amber-50/50')}>
                      <td className="sticky left-0 bg-background px-3 py-1.5">
                        <div className="flex items-center gap-1.5">
                          <div className="h-2 w-2 rounded-full" style={{ backgroundColor: getModelColor(m.model_type) }} />
                          <span className="text-xs font-semibold">{m.model_type.toUpperCase()}</span>
                          {comparison.best_model === m.model_type && <Trophy className="h-3 w-3 text-amber-500" />}
                        </div>
                      </td>
                      <td className="px-3 py-1.5 text-right text-xs">{fmt(m.rouge_1_f)}</td>
                      {showPR && <td className="px-2 py-1.5 text-right text-[10px] text-muted-foreground">{fmt(m.rouge_1_p)}</td>}
                      {showPR && <td className="px-2 py-1.5 text-right text-[10px] text-muted-foreground">{fmt(m.rouge_1_r)}</td>}
                      <td className="px-3 py-1.5 text-right text-xs">{fmt(m.rouge_2_f)}</td>
                      <td className="px-3 py-1.5 text-right text-xs">{fmt(m.rouge_l_f)}</td>
                      {showPR && <td className="px-2 py-1.5 text-right text-[10px] text-muted-foreground">{fmt(m.rouge_l_p)}</td>}
                      {showPR && <td className="px-2 py-1.5 text-right text-[10px] text-muted-foreground">{fmt(m.rouge_l_r)}</td>}
                      <td className="px-3 py-1.5 text-right text-xs">{fmt(m.bertscore_f1)}</td>
                      {showPR && <td className="px-2 py-1.5 text-right text-[10px] text-muted-foreground">{fmt(m.bertscore_precision)}</td>}
                      {showPR && <td className="px-2 py-1.5 text-right text-[10px] text-muted-foreground">{fmt(m.bertscore_recall)}</td>}
                      <td className="px-3 py-1.5 text-right text-xs">{fmt(m.factuality_score)}</td>
                      <td className="px-3 py-1.5 text-right text-xs">{fmt(m.semantic_similarity)}</td>
                      <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">{m.summary_length}</td>
                      <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">{m.generation_time?.toFixed(2)}s</td>
                      <td className="px-3 py-1.5 text-right text-xs font-semibold">{weightedScore(m).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {!hasEvals && (
            <div className="rounded-md border bg-muted/20 px-4 py-6 text-center">
              <AlertTriangle className="mx-auto h-6 w-6 text-muted-foreground" />
              <p className="mt-2 text-xs font-medium text-muted-foreground">No evaluation metrics computed yet</p>
              <p className="mt-0.5 text-[11px] text-muted-foreground">Click &quot;Evaluate All&quot; to run ROUGE, BERTScore, and factuality.</p>
            </div>
          )}

          {/* Charts */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-md border px-3 py-2">
              <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="Number of words in each generated summary">Summary Length</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={lengthData}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="model" tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip contentStyle={{fontSize:11}}/><Bar dataKey="words" fill="#3b82f6"/></BarChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-md border px-3 py-2">
              <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="Time to generate each summary">Generation Time (s)</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={lengthData}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="model" tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip contentStyle={{fontSize:11}}/><Bar dataKey="time" fill="#10b981"/></BarChart>
              </ResponsiveContainer>
            </div>
            {hasEvals && rougeData.length > 0 && (
              <div className="rounded-md border px-3 py-2">
                <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="ROUGE F1 scores across models">ROUGE Scores</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={rougeData}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="metric" tick={{fontSize:10}}/><YAxis domain={[0,1]} tick={{fontSize:10}}/><Tooltip contentStyle={{fontSize:11}} formatter={(v:number)=>v.toFixed(4)}/><Legend wrapperStyle={{fontSize:10}}/>
                    {models.map(m=><Bar key={m.model_type} dataKey={m.model_type.toUpperCase()} fill={getModelColor(m.model_type)}/>)}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {hasEvals && radarData.length > 0 && (
              <div className="rounded-md border px-3 py-2">
                <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground" title="Multi-metric radar comparison">Radar</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                    <PolarGrid/><PolarAngleAxis dataKey="metric" tick={{fontSize:9}}/><PolarRadiusAxis angle={30} domain={[0,100]} tick={{fontSize:8}}/>
                    {models.map(m=><Radar key={m.model_type} name={m.model_type.toUpperCase()} dataKey={m.model_type.toUpperCase()} stroke={getModelColor(m.model_type)} fill={getModelColor(m.model_type)} fillOpacity={0.15}/>)}
                    <Legend wrapperStyle={{fontSize:10}}/><Tooltip contentStyle={{fontSize:11}}/>
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Side-by-Side Summaries */}
          <div>
            <h3 className="mb-2 text-[11px] font-medium uppercase text-muted-foreground">Summary Text</h3>
            <div className="space-y-2">
              {models.map((m, i) => (
                <div key={m.summary_id ?? `${m.model_type}-${i}`} className="rounded-md border px-3 py-2" style={{ borderLeftWidth: '3px', borderLeftColor: getModelColor(m.model_type) }}>
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-semibold">{m.model_type.toUpperCase()}</span>
                    <Link href={`/summaries/${m.summary_id}`} className="text-[11px] text-primary hover:underline" title="View full summary details">Details</Link>
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">{m.summary_text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t bg-muted/30 px-4 py-2">
        <div className="grid grid-cols-5 gap-4">
          <div title="Number of models compared"><p className="text-[10px] font-medium uppercase text-muted-foreground">Models</p><p className="text-sm font-semibold">{models.length}</p></div>
          <div title="Original document word count"><p className="text-[10px] font-medium uppercase text-muted-foreground">Source</p><p className="text-sm font-semibold">{(comparison.word_count ?? 0).toLocaleString()} words</p></div>
          <div title="Document domain"><p className="text-[10px] font-medium uppercase text-muted-foreground">Domain</p><p className="text-sm font-semibold">{comparison.domain?.toUpperCase()}</p></div>
          <div title="Best performing model"><p className="text-[10px] font-medium uppercase text-muted-foreground">Best</p><p className="text-sm font-semibold">{comparison.best_model?.toUpperCase() || '—'}</p></div>
          <div title="Models with evaluation data"><p className="text-[10px] font-medium uppercase text-muted-foreground">Evaluated</p><p className="text-sm font-semibold">{models.filter(m=>m.rouge_l_f!==null).length}/{models.length}</p></div>
        </div>
      </div>
    </div>
  )
}
