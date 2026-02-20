'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import axios from 'axios'
import Link from 'next/link'
import { Loader2, ArrowLeft, Play, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Summary {
  id: number
  document_id: number
  model_type: string
  model_name: string
  summary_text: string
  summary_length: number
  generation_time: number
  generation_params: Record<string, any>
  created_at: string
}

interface Evaluation {
  id: number
  summary_id: number
  rouge_1_f: number | null
  rouge_1_p: number | null
  rouge_1_r: number | null
  rouge_2_f: number | null
  rouge_2_p: number | null
  rouge_2_r: number | null
  rouge_l_f: number | null
  rouge_l_p: number | null
  rouge_l_r: number | null
  bertscore_f1: number | null
  bertscore_precision: number | null
  bertscore_recall: number | null
  factuality_score: number | null
  factuality_method: string | null
  compression_ratio: number | null
  semantic_similarity: number | null
  evaluation_time: number | null
  evaluated_at: string
}

const modelBadge: Record<string, string> = {
  bart: 'warning',
  pegasus: 'secondary',
  gemini: 'info',
  gpt: 'success',
}

export default function SummaryDetailPage() {
  const params = useParams()
  const summaryId = params.id as string

  const [summary, setSummary] = useState<Summary | null>(null)
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [loading, setLoading] = useState(true)
  const [evaluating, setEvaluating] = useState(false)
  const [showPR, setShowPR] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get<Summary>(`${API_URL}/api/v1/summaries/${summaryId}`)
        setSummary(res.data)
        try {
          const evalRes = await axios.get<Evaluation>(`${API_URL}/api/v1/evaluations/summary/${summaryId}`)
          setEvaluation(evalRes.data)
        } catch {}
      } catch {}
      setLoading(false)
    }
    fetchData()
  }, [summaryId])

  const runEvaluation = async () => {
    setEvaluating(true)
    try {
      const res = await axios.post<Evaluation>(`${API_URL}/api/v1/evaluate/${summaryId}`)
      setEvaluation(res.data)
    } catch {
      alert('Evaluation failed. Check backend logs.')
    }
    setEvaluating(false)
  }

  const fmt = (v: number | null | undefined) => v !== null && v !== undefined ? v.toFixed(4) : '—'

  if (loading) {
    return (
      <div className="flex h-[calc(100dvh-64px)] flex-col">
        <div className="flex items-center gap-2 border-b bg-muted/30 px-4 py-2"><Skeleton className="h-4 w-4" /><Skeleton className="h-5 w-32" /></div>
        <div className="flex-1 space-y-4 px-4 py-4"><Skeleton className="h-40 w-full" /><Skeleton className="h-24 w-full" /></div>
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="flex h-[calc(100dvh-64px)] flex-col items-center justify-center">
        <p className="text-sm text-destructive">Summary not found</p>
        <Link href="/" className="mt-2 text-xs text-primary hover:underline">Go Home</Link>
      </div>
    )
  }

  const MetricRow = ({ label, f1, precision, recall, tooltip }: { label: string; f1: number | null; precision?: number | null; recall?: number | null; tooltip: string }) => {
    if (f1 === null && f1 === undefined) return null
    const pct = Math.min((f1 ?? 0) * 100, 100)
    return (
      <div title={tooltip}>
        <div className="flex items-center gap-3">
          <span className="w-28 text-xs text-muted-foreground">{label}</span>
          <div className="flex-1">
            <div className="h-1.5 w-full rounded-full bg-muted">
              <div className="h-1.5 rounded-full bg-primary transition-all duration-500" style={{ width: `${pct}%` }} />
            </div>
          </div>
          <span className="w-14 text-right text-xs font-medium">{fmt(f1)}</span>
        </div>
        {showPR && (precision !== null || recall !== null) && (
          <div className="ml-28 mt-0.5 flex gap-4 pl-1 text-[10px] text-muted-foreground">
            {precision !== null && precision !== undefined && <span>P: {fmt(precision)}</span>}
            {recall !== null && recall !== undefined && <span>R: {fmt(recall)}</span>}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100dvh-64px)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" asChild title="Back to document">
            <Link href={`/documents/${summary.document_id}`}><ArrowLeft className="h-3.5 w-3.5" /></Link>
          </Button>
          <Badge variant={(modelBadge[summary.model_type?.toLowerCase()] || 'secondary') as any}>
            {summary.model_type.toUpperCase()}
          </Badge>
          <span className="text-xs text-muted-foreground">Summary #{summary.id}</span>
        </div>
        <div className="flex items-center gap-2">
          {evaluation && (
            <button
              onClick={() => setShowPR(!showPR)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${showPR ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent'}`}
              title="Toggle Precision/Recall display"
            >
              {showPR ? 'Hide P/R' : 'Show P/R'}
            </button>
          )}
          {!evaluation && (
            <Button onClick={runEvaluation} disabled={evaluating} title="Run ROUGE, BERTScore, and factuality evaluation">
              {evaluating ? <><Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />Evaluating...</> : <><Play className="mr-1.5 h-3.5 w-3.5" />Run Evaluation</>}
            </Button>
          )}
          <Button variant="outline" asChild title="Compare this summary with other models">
            <Link href={`/compare/${summary.document_id}`}><BarChart3 className="mr-1.5 h-3.5 w-3.5" />Compare</Link>
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="mx-auto max-w-3xl space-y-6">
          {/* Info */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span title="Model used">{summary.model_name}</span>
            <span>&middot;</span>
            <span title="Generation time">{summary.generation_time?.toFixed(2)}s</span>
            <span>&middot;</span>
            <span title="Word count">{summary.summary_length} words</span>
            <span>&middot;</span>
            <span title="Generated at">{new Date(summary.created_at).toLocaleString()}</span>
          </div>

          {/* Summary Text */}
          <div>
            <h2 className="mb-2 text-xs font-medium uppercase text-muted-foreground">Summary Text</h2>
            <div className="rounded-md bg-muted/30 px-4 py-3">
              <p className="whitespace-pre-wrap text-xs leading-relaxed">{summary.summary_text}</p>
            </div>
          </div>

          {/* Evaluation Metrics */}
          <div>
            <h2 className="mb-2 text-xs font-medium uppercase text-muted-foreground">Evaluation Metrics</h2>
            {evaluation ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  <h3 className="text-[11px] font-medium text-muted-foreground">ROUGE Scores</h3>
                  <MetricRow label="ROUGE-1" f1={evaluation.rouge_1_f} precision={evaluation.rouge_1_p} recall={evaluation.rouge_1_r} tooltip="Unigram overlap between summary and source" />
                  <MetricRow label="ROUGE-2" f1={evaluation.rouge_2_f} precision={evaluation.rouge_2_p} recall={evaluation.rouge_2_r} tooltip="Bigram overlap between summary and source" />
                  <MetricRow label="ROUGE-L" f1={evaluation.rouge_l_f} precision={evaluation.rouge_l_p} recall={evaluation.rouge_l_r} tooltip="Longest common subsequence overlap" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-[11px] font-medium text-muted-foreground">Semantic Metrics</h3>
                  <MetricRow label="BERTScore F1" f1={evaluation.bertscore_f1} precision={evaluation.bertscore_precision} recall={evaluation.bertscore_recall} tooltip="Semantic similarity using contextual embeddings" />
                  <MetricRow label="Factuality" f1={evaluation.factuality_score} tooltip={`Factual consistency (method: ${evaluation.factuality_method || 'unknown'})`} />
                  <MetricRow label="Semantic Sim." f1={evaluation.semantic_similarity} tooltip="Contextual semantic alignment score" />
                </div>
                {(evaluation.compression_ratio !== null || evaluation.evaluation_time !== null) && (
                  <div className="flex gap-6 text-xs text-muted-foreground">
                    {evaluation.compression_ratio !== null && (
                      <span title="How much the summary reduces document length">Compression: <strong className="text-foreground">{evaluation.compression_ratio?.toFixed(3)}x</strong></span>
                    )}
                    {evaluation.evaluation_time !== null && (
                      <span title="Time taken to compute evaluation metrics">Eval time: <strong className="text-foreground">{evaluation.evaluation_time?.toFixed(2)}s</strong></span>
                    )}
                    {evaluation.factuality_method && (
                      <span title="Method used for factuality scoring">Factuality method: <strong className="text-foreground">{evaluation.factuality_method}</strong></span>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="py-8 text-center">
                <BarChart3 className="mx-auto h-8 w-8 text-muted-foreground/40" />
                <p className="mt-2 text-xs text-muted-foreground">No evaluation metrics yet</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">Click &quot;Run Evaluation&quot; to compute metrics.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t bg-muted/30 px-4 py-2">
        <div className="grid grid-cols-5 gap-4">
          <div title="Model used for this summary"><p className="text-[10px] font-medium uppercase text-muted-foreground">Model</p><p className="text-sm font-semibold">{summary.model_type.toUpperCase()}</p></div>
          <div title="Summary word count"><p className="text-[10px] font-medium uppercase text-muted-foreground">Words</p><p className="text-sm font-semibold">{summary.summary_length}</p></div>
          <div title="Time to generate"><p className="text-[10px] font-medium uppercase text-muted-foreground">Gen Time</p><p className="text-sm font-semibold">{summary.generation_time?.toFixed(2)}s</p></div>
          <div title="ROUGE-L F1 score"><p className="text-[10px] font-medium uppercase text-muted-foreground">ROUGE-L</p><p className="text-sm font-semibold">{fmt(evaluation?.rouge_l_f)}</p></div>
          <div title="BERTScore F1"><p className="text-[10px] font-medium uppercase text-muted-foreground">BERTScore</p><p className="text-sm font-semibold">{fmt(evaluation?.bertscore_f1)}</p></div>
        </div>
      </div>
    </div>
  )
}
