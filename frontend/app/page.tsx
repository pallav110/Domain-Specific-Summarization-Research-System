'use client'

import { useState } from 'react'
import Link from 'next/link'
import DocumentUploader from '@/components/DocumentUploader'
import DocumentList from '@/components/DocumentList'
import { FileText, BarChart3, FlaskConical } from 'lucide-react'

export default function Home() {
  const [uploadedDocId, setUploadedDocId] = useState<number | null>(null)

  const handleUploadSuccess = (docId: number) => {
    setUploadedDocId(docId)
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center py-12 bg-gradient-to-r from-primary-600 to-primary-800 text-white rounded-2xl shadow-xl">
        <h1 className="text-5xl font-bold mb-4">
          Domain-Specific Summarization Research
        </h1>
        <p className="text-xl text-primary-100 max-w-3xl mx-auto">
          Comparing Generic vs Domain-Specific NLP Models for Legal and Medical Document Summarization
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link href="/documents" className="card hover:shadow-lg transition-shadow cursor-pointer">
          <div className="flex items-center space-x-4">
            <div className="bg-primary-100 p-3 rounded-lg">
              <FileText className="w-8 h-8 text-primary-600" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-800">Documents</h3>
              <p className="text-gray-600">Upload & Process</p>
            </div>
          </div>
        </Link>

        <Link href="/experiments" className="card hover:shadow-lg transition-shadow cursor-pointer">
          <div className="flex items-center space-x-4">
            <div className="bg-green-100 p-3 rounded-lg">
              <FlaskConical className="w-8 h-8 text-green-600" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-800">Experiments</h3>
              <p className="text-gray-600">Run & Compare</p>
            </div>
          </div>
        </Link>

        <Link href="/dashboard" className="card hover:shadow-lg transition-shadow cursor-pointer">
          <div className="flex items-center space-x-4">
            <div className="bg-purple-100 p-3 rounded-lg">
              <BarChart3 className="w-8 h-8 text-purple-600" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-800">Dashboard</h3>
              <p className="text-gray-600">View Results</p>
            </div>
          </div>
        </Link>
      </div>

      {/* Upload Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">Upload Document</h2>
          <p className="text-gray-600 mb-6">
            Upload a legal or medical document to start the summarization process
          </p>
          <DocumentUploader onUploadSuccess={handleUploadSuccess} />
        </div>

        <div className="card">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">Research Focus</h2>
          <div className="space-y-4 text-gray-700">
            <div className="flex items-start space-x-3">
              <div className="bg-primary-100 rounded-full p-1 mt-1">
                <div className="w-2 h-2 bg-primary-600 rounded-full"></div>
              </div>
              <p>Compare generic models (BART, PEGASUS) with domain-specific models (Legal-BERT, Clinical-BERT)</p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-primary-100 rounded-full p-1 mt-1">
                <div className="w-2 h-2 bg-primary-600 rounded-full"></div>
              </div>
              <p>Evaluate using ROUGE, BERTScore, and factuality metrics</p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-primary-100 rounded-full p-1 mt-1">
                <div className="w-2 h-2 bg-primary-600 rounded-full"></div>
              </div>
              <p>Test GPT-4 as a baseline for state-of-the-art performance</p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-primary-100 rounded-full p-1 mt-1">
                <div className="w-2 h-2 bg-primary-600 rounded-full"></div>
              </div>
              <p>Generate publication-ready experimental results</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Documents */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Recent Documents</h2>
        <DocumentList limit={5} highlightId={uploadedDocId} />
      </div>
    </div>
  )
}
