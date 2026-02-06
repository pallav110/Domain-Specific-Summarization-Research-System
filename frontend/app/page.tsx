'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Crimson_Pro, Source_Sans_3 } from 'next/font/google'
import DocumentUploader from '@/components/DocumentUploader'
import DocumentList from '@/components/DocumentList'
import {
  FileText,
  BarChart3,
  FlaskConical,
  Scale,
  Stethoscope,
  UploadCloud
} from 'lucide-react'

const serif = Crimson_Pro({
  subsets: ['latin'],
  weight: ['400', '600', '700']
})

const sans = Source_Sans_3({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700']
})

export default function Home() {
  const [uploadedDocId, setUploadedDocId] = useState<number | null>(null)

  const handleUploadSuccess = (docId: number) => {
    setUploadedDocId(docId)
  }

  return (
    <div className={`${sans.className} space-y-10`}>
        {/* Hero Section */}
        <section className="rounded-2xl border border-slate-200 bg-white">
          <div className="grid gap-10 px-8 py-10 md:grid-cols-[1.1fr_0.9fr] md:px-10">
            <div className="space-y-6">
              <h1 className={`${serif.className} text-4xl font-semibold leading-tight text-slate-900 md:text-5xl`}>
                Domain-Specific Summarization Research
              </h1>
              <p className="text-base text-slate-600 md:text-lg">
                A structured environment for comparing generic and domain-specific models on legal
                and medical documents with clear, measurable outcomes.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  href="/documents"
                  className="btn-primary"
                >
                  Upload Documents
                </Link>
                <Link
                  href="/experiments"
                  className="btn-secondary"
                >
                  Run Experiments
                </Link>
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-6">
              <h3 className={`${serif.className} text-lg font-semibold text-slate-900`}>Research Snapshot</h3>
              <div className="mt-5 space-y-4 text-sm text-slate-600">
                <div className="flex items-start gap-3">
                  <Scale className="h-5 w-5 text-blue-700" />
                  Compare BART, PEGASUS, Gemini, Legal-BERT+PEGASUS, and Clinical-BERT+PEGASUS.
                </div>
                <div className="flex items-start gap-3">
                  <Stethoscope className="h-5 w-5 text-blue-700" />
                  Track fidelity with ROUGE, BERTScore, factuality, and semantic similarity.
                </div>
                <div className="flex items-start gap-3">
                  <FlaskConical className="h-5 w-5 text-blue-700" />
                  Export CSV/JSON for external analysis and reporting.
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Quick Links */}
        <section className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <Link href="/documents" className="card transition-shadow hover:border-slate-300">
            <div className="flex items-center gap-4">
              <div className="rounded-xl bg-blue-50 p-3 text-blue-700">
                <FileText className="h-8 w-8" />
              </div>
              <div>
                <h3 className={`${serif.className} text-2xl font-semibold text-slate-900`}>Documents</h3>
                <p className="text-slate-600">Upload and classify files</p>
              </div>
            </div>
          </Link>

          <Link href="/experiments" className="card transition-shadow hover:border-slate-300">
            <div className="flex items-center gap-4">
              <div className="rounded-xl bg-slate-100 p-3 text-slate-700">
                <FlaskConical className="h-8 w-8" />
              </div>
              <div>
                <h3 className={`${serif.className} text-2xl font-semibold text-slate-900`}>Experiments</h3>
                <p className="text-slate-600">Run structured comparisons</p>
              </div>
            </div>
          </Link>

          <Link href="/dashboard" className="card transition-shadow hover:border-slate-300">
            <div className="flex items-center gap-4">
              <div className="rounded-xl bg-slate-100 p-3 text-slate-700">
                <BarChart3 className="h-8 w-8" />
              </div>
              <div>
                <h3 className={`${serif.className} text-2xl font-semibold text-slate-900`}>Dashboard</h3>
                <p className="text-slate-600">Review metrics and trends</p>
              </div>
            </div>
          </Link>
        </section>

        {/* Upload + Focus */}
        <section className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <div className="card">
            <div className="flex items-center justify-between">
              <h2 className={`${serif.className} text-2xl font-semibold text-slate-900`}>Upload Document</h2>
              <div className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
                <UploadCloud className="h-4 w-4" />
                PDF or TXT
              </div>
            </div>
            <p className="mt-2 text-sm text-slate-600">
              Submit a legal or medical document and generate summaries across all models.
            </p>
            <div className="mt-6">
              <DocumentUploader onUploadSuccess={handleUploadSuccess} />
            </div>
          </div>

          <div className="card">
            <h2 className={`${serif.className} text-2xl font-semibold text-slate-900`}>Research Focus</h2>
            <div className="mt-4 space-y-4 text-sm text-slate-700">
              <div className="flex items-start gap-3">
                <div className="mt-1 h-2 w-2 rounded-full bg-blue-700" />
                Compare generic models (BART, PEGASUS) with domain-specific pipelines (Legal-BERT+PEGASUS, Clinical-BERT+PEGASUS).
              </div>
              <div className="flex items-start gap-3">
                <div className="mt-1 h-2 w-2 rounded-full bg-blue-700" />
                Evaluate summaries using ROUGE, BERTScore, factuality, semantic similarity, and compression rate.
              </div>
              <div className="flex items-start gap-3">
                <div className="mt-1 h-2 w-2 rounded-full bg-blue-700" />
                Use Gemini as an LLM baseline for quality and coverage.
              </div>
              <div className="flex items-start gap-3">
                <div className="mt-1 h-2 w-2 rounded-full bg-blue-700" />
                Export results for statistical analysis and publication-ready tables.
              </div>
            </div>
          </div>
        </section>

        {/* Recent Documents */}
        <section className="card">
          <div className="flex items-center justify-between">
            <h2 className={`${serif.className} text-2xl font-semibold text-slate-900`}>Recent Documents</h2>
            <Link href="/documents" className="text-sm font-semibold text-blue-700 hover:text-blue-800">
              View all
            </Link>
          </div>
          <div className="mt-6">
            <DocumentList limit={5} highlightId={uploadedDocId} />
          </div>
        </section>
    </div>
  )
}
