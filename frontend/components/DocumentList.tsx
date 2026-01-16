'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import Link from 'next/link'
import { FileText, Calendar, Hash, Loader2 } from 'lucide-react'

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

export default function DocumentList({ limit, highlightId }: Props) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchDocuments()
  }, [limit])

  const fetchDocuments = async () => {
    try {
      const params = limit ? { limit } : {}
      const response = await axios.get<Document[]>(
        `${API_URL}/api/v1/documents`,
        { params }
      )
      setDocuments(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <Loader2 className="w-8 h-8 text-primary-600 animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12 text-red-600">
        <p>{error}</p>
      </div>
    )
  }

  if (documents.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <FileText className="w-12 h-12 mx-auto mb-3 text-gray-400" />
        <p>No documents uploaded yet</p>
      </div>
    )
  }

  const getDomainColor = (domain: string) => {
    switch (domain.toLowerCase()) {
      case 'legal':
        return 'bg-blue-100 text-blue-800'
      case 'medical':
        return 'bg-green-100 text-green-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-3">
      {documents.map((doc) => (
        <Link
          key={doc.id}
          href={`/documents/${doc.id}`}
          className={`block p-4 rounded-lg border transition-all hover:shadow-md ${
            highlightId === doc.id
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-200 hover:border-primary-300'
          }`}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3 flex-1">
              <FileText className="w-5 h-5 text-gray-400 mt-1 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <h3 className="font-medium text-gray-900 truncate">
                  {doc.original_filename}
                </h3>
                <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                  <span className="flex items-center space-x-1">
                    <Hash className="w-4 h-4" />
                    <span>{doc.word_count.toLocaleString()} words</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <Calendar className="w-4 h-4" />
                    <span>{new Date(doc.upload_timestamp).toLocaleDateString()}</span>
                  </span>
                </div>
              </div>
            </div>
            <div className="ml-4 flex flex-col items-end space-y-2">
              <span
                className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getDomainColor(
                  doc.detected_domain
                )}`}
              >
                {doc.detected_domain.toUpperCase()}
              </span>
              <span className="text-xs text-gray-500">
                {(doc.domain_confidence * 100).toFixed(0)}% confidence
              </span>
            </div>
          </div>
        </Link>
      ))}
    </div>
  )
}
