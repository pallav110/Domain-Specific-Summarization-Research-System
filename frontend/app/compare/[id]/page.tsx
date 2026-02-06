'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import axios from 'axios'
import Link from 'next/link'
import {
  Loader2, ArrowLeft, BarChart3, Trophy, Clock, Hash, AlertTriangle
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Summary {
  id: number
  document_id: number
  model_type: string
  model_name: string
  summary_text: string
  summary_length: number
  generation_time: number
  created_at: string
}

interface Document {
  id: number
  original_filename: string
  word_count: number
  detected_domain: string
  domain_confidence: number
  raw_text: string
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
}

interface ModelData {
  summary: Summary
  evaluation: Evaluation | null
}

const MODEL_COLORS: Record<string, string> = {
  bart: '#f97316',
  pegasus: '#8b5cf6',
  gemini: '#3b82f6',
  gpt: '#10b981',
  t5: '#ef4444',
  legal_bert_pegasus: '#06b6d4',
  clinical_bert_pegasus: '#ec4899',
}

export default function ComparePage() {
  const params = useParams()
  const documentId = params.id as string

  const [document, setDocument] = useState<Document | null>(null)
  const [models, setModels] = useState<ModelData[]>([])
  const [loading, setLoading] = useState(true)
  const [evaluatingAll, setEvaluatingAll] = useState(false)

  useEffect(() => {
    fetchData()
  }, [documentId])

  const fetchData = async () => {
    try {
      const [docRes, sumRes] = await Promise.all([
        axios.get<Document>(`${API_URL}/api/v1/documents/${documentId}`),
        axios.get<Summary[]>(`${API_URL}/api/v1/documents/${documentId}/summaries`),
      ])
      setDocument(docRes.data)

      // Try to fetch evaluations for each summary
      const modelData: ModelData[] = await Promise.all(
        sumRes.data.map(async (summary) => {
          let evaluation: Evaluation | null = null
          try {
            const evalRes = await axios.get<Evaluation>(
              `${API_URL}/api/v1/evaluations/summary/${summary.id}`
            )
            evaluation = evalRes.data
          } catch {
            // No evaluation yet
          }
          return { summary, evaluation }
        })
      )
      setModels(modelData)
    } catch (err) {
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const evaluateAll = async () => {
    setEvaluatingAll(true)
    try {
      const updated = await Promise.all(
        models.map(async (m) => {
          if (m.evaluation) return m
          try {
            const res = await axios.post<Evaluation>(
              `${API_URL}/api/v1/evaluate/${m.summary.id}`
            )
            return { ...m, evaluation: res.data }
          } catch {
            return m
          }
        })
      )
      setModels(updated)
    } catch (err) {
      console.error('Evaluation failed:', err)
    } finally {
      setEvaluatingAll(false)
    }
  }

  const getModelColor = (model: string) => MODEL_COLORS[model?.toLowerCase()] || '#6b7280'

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
      </div>
    )
  }

  if (!document || models.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 text-lg">No summaries found for this document.</p>
        <Link href={`/documents/${documentId}`} className="text-primary-600 hover:underline mt-2 inline-block">
          Go back and generate summaries
        </Link>
      </div>
    )
  }

  // Prepare chart data
  const hasEvals = models.some(m => m.evaluation)

  const lengthData = models.map(m => ({
    model: m.summary.model_type.toUpperCase(),
    words: m.summary.summary_length,
    time: parseFloat(m.summary.generation_time?.toFixed(2) || '0'),
    fill: getModelColor(m.summary.model_type),
  }))

  const compressionData = models.map(m => ({
    model: m.summary.model_type.toUpperCase(),
    ratio: document.word_count > 0
      ? parseFloat(((1 - m.summary.summary_length / document.word_count) * 100).toFixed(1))
      : 0,
    fill: getModelColor(m.summary.model_type),
  }))

  const rougeData = hasEvals
    ? [
        {
          metric: 'ROUGE-1',
          ...Object.fromEntries(models.map(m => [
            m.summary.model_type.toUpperCase(),
            m.evaluation?.rouge_1_f ?? 0,
          ])),
        },
        {
          metric: 'ROUGE-2',
          ...Object.fromEntries(models.map(m => [
            m.summary.model_type.toUpperCase(),
            m.evaluation?.rouge_2_f ?? 0,
          ])),
        },
        {
          metric: 'ROUGE-L',
          ...Object.fromEntries(models.map(m => [
            m.summary.model_type.toUpperCase(),
            m.evaluation?.rouge_l_f ?? 0,
          ])),
        },
      ]
    : []

  const radarData = hasEvals
    ? [
        { metric: 'ROUGE-1', ...Object.fromEntries(models.map(m => [m.summary.model_type.toUpperCase(), (m.evaluation?.rouge_1_f ?? 0) * 100])) },
        { metric: 'ROUGE-2', ...Object.fromEntries(models.map(m => [m.summary.model_type.toUpperCase(), (m.evaluation?.rouge_2_f ?? 0) * 100])) },
        { metric: 'ROUGE-L', ...Object.fromEntries(models.map(m => [m.summary.model_type.toUpperCase(), (m.evaluation?.rouge_l_f ?? 0) * 100])) },
        { metric: 'BERTScore', ...Object.fromEntries(models.map(m => [m.summary.model_type.toUpperCase(), (m.evaluation?.bertscore_f1 ?? 0) * 100])) },
        { metric: 'Factuality', ...Object.fromEntries(models.map(m => [m.summary.model_type.toUpperCase(), (m.evaluation?.factuality_score ?? 0) * 100])) },
      ]
    : []

  // Find best model per metric
  const findBest = (getter: (e: Evaluation) => number | null) => {
    let best: ModelData | null = null
    let bestVal = -1
    for (const m of models) {
      const v = m.evaluation ? getter(m.evaluation) : null
      if (v !== null && v > bestVal) {
        bestVal = v
        best = m
      }
    }
    return best?.summary.model_type.toUpperCase()
  }

  return (
    <div className="space-y-6">
      <Link
        href={`/documents/${documentId}`}
        className="inline-flex items-center space-x-2 text-primary-600 hover:text-primary-700"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Document</span>
      </Link>

      {/* Header */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-2">
          <Trophy className="w-8 h-8 text-yellow-500" />
          <h1 className="text-3xl font-bold text-gray-800">Model Comparison</h1>
        </div>
        <p className="text-gray-600">
          Comparing {models.length} models on{' '}
          <span className="font-semibold">{document.original_filename}</span>
          {' '}({document.word_count} words, {document.detected_domain.toUpperCase()} domain)
        </p>
      </div>

      {/* Quick Stats + Evaluate All */}
      <div className="flex flex-wrap gap-4 items-center">
        {models.map(m => (
          <div key={m.summary.id} className="card flex-1 min-w-[200px]" style={{ borderTop: `4px solid ${getModelColor(m.summary.model_type)}` }}>
            <h3 className="font-bold text-lg">{m.summary.model_type.toUpperCase()}</h3>
            <div className="text-sm text-gray-500 space-y-1 mt-2">
              <div className="flex items-center space-x-1">
                <Hash className="w-3 h-3" />
                <span>{m.summary.summary_length} words</span>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="w-3 h-3" />
                <span>{m.summary.generation_time?.toFixed(2)}s</span>
              </div>
              <div>
                Compression: {((1 - m.summary.summary_length / document.word_count) * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ))}
      </div>

      {!hasEvals && (
        <div className="card bg-yellow-50 border border-yellow-200">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <p className="font-medium text-yellow-800">No evaluation metrics computed yet</p>
              <p className="text-sm text-yellow-700 mt-1">
                Run evaluation to see ROUGE, BERTScore, and factuality comparisons.
              </p>
              <button
                onClick={evaluateAll}
                disabled={evaluatingAll}
                className="mt-3 bg-yellow-600 text-white px-4 py-2 rounded-lg hover:bg-yellow-700 disabled:opacity-50 text-sm inline-flex items-center space-x-2"
              >
                {evaluatingAll ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /><span>Evaluating All...</span></>
                ) : (
                  <><BarChart3 className="w-4 h-4" /><span>Evaluate All Models</span></>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Word Count Comparison */}
        <div className="card">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Summary Length (words)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={lengthData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="words" fill="#3b82f6">
                {lengthData.map((entry, i) => (
                  <Bar key={i} dataKey="words" fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Generation Time */}
        <div className="card">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Generation Time (seconds)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={lengthData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="time" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Compression Ratio */}
        <div className="card">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Compression Rate (%)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={compressionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(v: number) => `${v}%`} />
              <Bar dataKey="ratio" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* ROUGE Scores */}
        {hasEvals && rougeData.length > 0 && (
          <div className="card">
            <h2 className="text-lg font-bold mb-4 text-gray-800">ROUGE Scores</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={rougeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(v: number) => v.toFixed(4)} />
                <Legend />
                {models.map(m => (
                  <Bar
                    key={m.summary.model_type}
                    dataKey={m.summary.model_type.toUpperCase()}
                    fill={getModelColor(m.summary.model_type)}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Radar Chart */}
      {hasEvals && radarData.length > 0 && (
        <div className="card">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Multi-Metric Radar</h2>
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis angle={30} domain={[0, 100]} />
              {models.map(m => (
                <Radar
                  key={m.summary.model_type}
                  name={m.summary.model_type.toUpperCase()}
                  dataKey={m.summary.model_type.toUpperCase()}
                  stroke={getModelColor(m.summary.model_type)}
                  fill={getModelColor(m.summary.model_type)}
                  fillOpacity={0.15}
                />
              ))}
              <Legend />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Best Model Summary */}
      {hasEvals && (
        <div className="card bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200">
          <h2 className="text-lg font-bold mb-3 text-gray-800 flex items-center space-x-2">
            <Trophy className="w-5 h-5 text-yellow-500" />
            <span>Best Model by Metric</span>
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { label: 'ROUGE-1', best: findBest(e => e.rouge_1_f) },
              { label: 'ROUGE-2', best: findBest(e => e.rouge_2_f) },
              { label: 'ROUGE-L', best: findBest(e => e.rouge_l_f) },
              { label: 'BERTScore', best: findBest(e => e.bertscore_f1) },
              { label: 'Factuality', best: findBest(e => e.factuality_score) },
              { label: 'Semantic Sim.', best: findBest(e => e.semantic_similarity) },
            ].map(item => (
              <div key={item.label} className="bg-white rounded-lg p-3 shadow-sm">
                <p className="text-xs text-gray-500">{item.label}</p>
                <p className="font-bold text-gray-800">{item.best || 'N/A'}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Side-by-Side Summaries */}
      <div className="card">
        <h2 className="text-lg font-bold mb-4 text-gray-800">Side-by-Side Summary Text</h2>
        <div className="space-y-4">
          {models.map(m => (
            <div key={m.summary.id} className="border rounded-lg p-4" style={{ borderLeftWidth: '4px', borderLeftColor: getModelColor(m.summary.model_type) }}>
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm">{m.summary.model_type.toUpperCase()}</span>
                <Link
                  href={`/summaries/${m.summary.id}`}
                  className="text-xs text-primary-600 hover:underline"
                >
                  Full Details →
                </Link>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">
                {m.summary.summary_text}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
