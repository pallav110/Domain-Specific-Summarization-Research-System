'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import axios from 'axios'
import Link from 'next/link'
import { Loader2, ArrowLeft, FileText, Clock, Hash, BarChart3, Play } from 'lucide-react'

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
  rouge_2_f: number | null
  rouge_l_f: number | null
  bertscore_f1: number | null
  factuality_score: number | null
  compression_ratio: number | null
  semantic_similarity: number | null
  evaluated_at: string
}

export default function SummaryDetailPage() {
  const params = useParams()
  const summaryId = params.id as string

  const [summary, setSummary] = useState<Summary | null>(null)
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [loading, setLoading] = useState(true)
  const [evaluating, setEvaluating] = useState(false)

  useEffect(() => {
    fetchSummary()
  }, [summaryId])

  const fetchSummary = async () => {
    try {
      const res = await axios.get<Summary>(`${API_URL}/api/v1/summaries/${summaryId}`)
      setSummary(res.data)

      // Try to fetch existing evaluation
      try {
        const evalRes = await axios.get<Evaluation>(
          `${API_URL}/api/v1/evaluations/summary/${summaryId}`
        )
        setEvaluation(evalRes.data)
      } catch {
        // No evaluation yet
      }
    } catch (err) {
      console.error('Failed to fetch summary:', err)
    } finally {
      setLoading(false)
    }
  }

  const runEvaluation = async () => {
    setEvaluating(true)
    try {
      const res = await axios.post<Evaluation>(
        `${API_URL}/api/v1/evaluate/${summaryId}`
      )
      setEvaluation(res.data)
    } catch (err) {
      console.error('Evaluation failed:', err)
      alert('Evaluation failed. This may require reference text or additional setup. Check backend logs.')
    } finally {
      setEvaluating(false)
    }
  }

  const getModelColor = (model: string) => {
    switch (model?.toLowerCase()) {
      case 'bart': return 'bg-orange-100 text-orange-800'
      case 'pegasus': return 'bg-purple-100 text-purple-800'
      case 'gemini': return 'bg-blue-100 text-blue-800'
      case 'gpt': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const MetricBar = ({ label, value, max = 1 }: { label: string; value: number | null; max?: number }) => {
    if (value === null || value === undefined) return null
    const pct = Math.min((value / max) * 100, 100)
    return (
      <div className="space-y-1">
        <div className="flex justify-between text-sm">
          <span className="font-medium text-gray-700">{label}</span>
          <span className="text-gray-600">{value.toFixed(4)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className="bg-primary-600 h-2.5 rounded-full transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 text-lg">Summary not found</p>
        <Link href="/" className="text-primary-600 hover:underline mt-2 inline-block">
          Go Home
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Back */}
      <Link
        href={`/documents/${summary.document_id}`}
        className="inline-flex items-center space-x-2 text-primary-600 hover:text-primary-700"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Document</span>
      </Link>

      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getModelColor(summary.model_type)}`}>
                {summary.model_type.toUpperCase()}
              </span>
              <span className="text-sm text-gray-500">Summary #{summary.id}</span>
            </div>
            <p className="text-sm text-gray-600">{summary.model_name}</p>
          </div>
          <div className="text-right text-sm text-gray-500 space-y-1">
            <div className="flex items-center space-x-1 justify-end">
              <Clock className="w-4 h-4" />
              <span>{summary.generation_time?.toFixed(2)}s</span>
            </div>
            <div className="flex items-center space-x-1 justify-end">
              <Hash className="w-4 h-4" />
              <span>{summary.summary_length} words</span>
            </div>
            <div className="text-xs">
              {new Date(summary.created_at).toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      {/* Summary Text */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4 text-gray-800 flex items-center space-x-2">
          <FileText className="w-5 h-5" />
          <span>Summary Text</span>
        </h2>
        <div className="bg-gray-50 rounded-lg p-6">
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
            {summary.summary_text}
          </p>
        </div>
      </div>

      {/* Evaluation Metrics */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-800 flex items-center space-x-2">
            <BarChart3 className="w-5 h-5" />
            <span>Evaluation Metrics</span>
          </h2>
          {!evaluation && (
            <button
              onClick={runEvaluation}
              disabled={evaluating}
              className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 text-sm inline-flex items-center space-x-2"
            >
              {evaluating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Run Evaluation</span>
                </>
              )}
            </button>
          )}
        </div>

        {evaluation ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">
                ROUGE Scores
              </h3>
              <MetricBar label="ROUGE-1 (Unigram)" value={evaluation.rouge_1_f} />
              <MetricBar label="ROUGE-2 (Bigram)" value={evaluation.rouge_2_f} />
              <MetricBar label="ROUGE-L (Longest Common)" value={evaluation.rouge_l_f} />
            </div>
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">
                Semantic Metrics
              </h3>
              <MetricBar label="BERTScore F1" value={evaluation.bertscore_f1} />
              <MetricBar label="Factuality Score" value={evaluation.factuality_score} />
              <MetricBar label="Semantic Similarity" value={evaluation.semantic_similarity} />
              {evaluation.compression_ratio !== null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-gray-700">Compression Ratio</span>
                    <span className="text-gray-600">{evaluation.compression_ratio?.toFixed(2)}x</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center py-8 bg-gray-50 rounded-lg">
            <BarChart3 className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No evaluation metrics yet.</p>
            <p className="text-gray-400 text-sm mt-1">
              Click &quot;Run Evaluation&quot; to compute ROUGE, BERTScore, and factuality metrics.
            </p>
          </div>
        )}
      </div>

      {/* Compare Link */}
      <div className="card text-center">
        <Link
          href={`/compare/${summary.document_id}`}
          className="btn-primary inline-flex items-center space-x-2"
        >
          <span>Compare with Other Models</span>
          <span>→</span>
        </Link>
      </div>
    </div>
  )
}
