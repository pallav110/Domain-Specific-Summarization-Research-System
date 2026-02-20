'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { Upload, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface UploadResult {
  document_id: number
  filename: string
  file_size: number
  word_count: number
  detected_domain: string
  domain_confidence: number
  message: string
}

interface Props {
  onUploadSuccess?: (documentId: number) => void
}

export default function DocumentUploader({ onUploadSuccess }: Props) {
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    const file = acceptedFiles[0]
    setUploading(true)
    setError(null)
    setUploadResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post<UploadResult>(
        `${API_URL}/api/v1/documents/upload`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setUploadResult(response.data)
      onUploadSuccess?.(response.data.document_id)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [onUploadSuccess])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt'] },
    maxFiles: 1,
    disabled: uploading,
  })

  return (
    <div className="space-y-2">
      <div
        {...getRootProps()}
        className={cn(
          'cursor-pointer rounded-md border border-dashed p-6 text-center transition-colors',
          isDragActive ? 'border-primary bg-primary/5' : 'border-input hover:border-primary/50',
          uploading && 'pointer-events-none opacity-50'
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-1.5">
          {uploading ? (
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          ) : (
            <Upload className="h-6 w-6 text-muted-foreground" />
          )}
          <p className="text-xs font-medium">
            {isDragActive ? 'Drop file here' : 'Drag & drop or click to select'}
          </p>
          <p className="text-[11px] text-muted-foreground">PDF or TXT</p>
        </div>
      </div>

      {uploadResult && (
        <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2">
          <div className="flex items-start gap-2">
            <CheckCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
            <div className="min-w-0 flex-1 text-xs">
              <p className="font-medium text-emerald-900">Uploaded successfully</p>
              <p className="mt-0.5 text-emerald-700">
                {uploadResult.filename} &middot; {uploadResult.word_count.toLocaleString()} words &middot; {uploadResult.detected_domain.toUpperCase()} ({(uploadResult.domain_confidence * 100).toFixed(0)}%)
              </p>
              <a
                href={`/documents/${uploadResult.document_id}`}
                className="mt-1 inline-block text-[11px] font-medium text-primary hover:underline"
              >
                View details
              </a>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2">
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600" />
            <div className="text-xs">
              <p className="font-medium text-red-900">Upload failed</p>
              <p className="mt-0.5 text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
