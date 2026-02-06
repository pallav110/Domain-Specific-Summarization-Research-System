'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Loader2, BarChart3, TrendingUp, Award, Download } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
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

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get<DashboardStats>(
        `${API_URL}/api/v1/dashboard/stats`
      )
      setStats(response.data)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="text-center py-12 text-red-600">
        <p>Failed to load dashboard data</p>
      </div>
    )
  }

  // Prepare chart data
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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <BarChart3 className="w-8 h-8 text-primary-600" />
        <h1 className="text-3xl font-bold text-gray-800">Research Dashboard</h1>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card bg-gradient-to-br from-blue-50 to-blue-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 font-medium">Total Documents</p>
              <p className="text-3xl font-bold text-blue-900 mt-1">
                {stats.total_documents}
              </p>
            </div>
            <div className="bg-blue-200 p-3 rounded-lg">
              <BarChart3 className="w-8 h-8 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="card bg-gradient-to-br from-green-50 to-green-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 font-medium">Total Summaries</p>
              <p className="text-3xl font-bold text-green-900 mt-1">
                {stats.total_summaries}
              </p>
            </div>
            <div className="bg-green-200 p-3 rounded-lg">
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
          </div>
        </div>

        <div className="card bg-gradient-to-br from-purple-50 to-purple-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600 font-medium">Experiments</p>
              <p className="text-3xl font-bold text-purple-900 mt-1">
                {stats.total_experiments}
              </p>
            </div>
            <div className="bg-purple-200 p-3 rounded-lg">
              <Award className="w-8 h-8 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Domain Distribution */}
        <div className="card">
          <h2 className="text-xl font-bold mb-4 text-gray-800">Domain Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={domainData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {domainData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Model Usage */}
        <div className="card">
          <h2 className="text-xl font-bold mb-4 text-gray-800">Model Usage</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Average Metrics */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4 text-gray-800">Average Evaluation Metrics</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={metricsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(value: number) => value.toFixed(4)} />
            <Legend />
            <Bar dataKey="score" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Research Insights */}
      <div className="card bg-gradient-to-r from-primary-50 to-purple-50">
        <h2 className="text-xl font-bold mb-4 text-gray-800">Research Insights</h2>
        <div className="space-y-3 text-gray-700">
          <div className="flex items-start space-x-3">
            <div className="bg-primary-600 rounded-full p-1 mt-1">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
            <p>
              <strong>Average ROUGE-L:</strong>{' '}
              {(stats.average_metrics.rouge_l || 0).toFixed(4)} - Measures content overlap
              between summary and source
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="bg-primary-600 rounded-full p-1 mt-1">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
            <p>
              <strong>Average BERTScore:</strong>{' '}
              {(stats.average_metrics.bertscore || 0).toFixed(4)} - Measures semantic
              similarity using contextual embeddings
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="bg-primary-600 rounded-full p-1 mt-1">
              <div className="w-2 h-2 bg-white rounded-full"></div>
            </div>
            <p>
              <strong>Average Factuality:</strong>{' '}
              {(stats.average_metrics.factuality || 0).toFixed(4)} - Measures factual
              consistency with source document
            </p>
          </div>
        </div>
      </div>

      {/* Export Section */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4 text-gray-800">Export Research Data</h2>
        <p className="text-sm text-gray-600 mb-4">
          Download evaluation results for external analysis in Python, R, or Excel.
        </p>
        <div className="flex flex-wrap gap-4">
          <a
            href={`${API_URL}/api/v1/export/csv`}
            className="btn-primary inline-flex items-center space-x-2"
            download
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </a>
          <a
            href={`${API_URL}/api/v1/export/json`}
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 inline-flex items-center space-x-2"
            download
          >
            <Download className="w-4 h-4" />
            <span>Export JSON</span>
          </a>
          <a
            href={`${API_URL}/api/v1/statistics/analysis`}
            target="_blank"
            rel="noopener noreferrer"
            className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 inline-flex items-center space-x-2"
          >
            <TrendingUp className="w-4 h-4" />
            <span>View Statistical Analysis</span>
          </a>
        </div>
      </div>
    </div>
  )
}
