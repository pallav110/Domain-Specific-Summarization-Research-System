'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { Upload, FileText, Loader2, CheckCircle, AlertCircle } from 'lucide-react'

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
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )

      setUploadResult(response.data)
      if (onUploadSuccess) {
        onUploadSuccess(response.data.document_id)
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [onUploadSuccess])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
    disabled: uploading,
  })

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400'
        } ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-3">
          {uploading ? (
            <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
          ) : (
            <Upload className="w-12 h-12 text-gray-400" />
          )}
          
          <div>
            <p className="text-lg font-medium text-gray-700">
              {isDragActive ? 'Drop the file here' : 'Drag & drop a document'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to select (PDF or TXT)
            </p>
          </div>
        </div>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-medium text-green-900 mb-2">Upload Successful!</h3>
              <div className="text-sm text-green-800 space-y-1">
                <p><strong>File:</strong> {uploadResult.filename}</p>
                <p><strong>Size:</strong> {(uploadResult.file_size / 1024).toFixed(2)} KB</p>
                <p><strong>Words:</strong> {uploadResult.word_count.toLocaleString()}</p>
                <p>
                  <strong>Domain:</strong>{' '}
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                    {uploadResult.detected_domain.toUpperCase()}
                  </span>
                  {' '}({(uploadResult.domain_confidence * 100).toFixed(1)}% confidence)
                </p>
              </div>
              <a
                href={`/documents/${uploadResult.document_id}`}
                className="inline-block mt-3 text-sm font-medium text-primary-600 hover:text-primary-700"
              >
                View Document Details →
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-medium text-red-900">Upload Failed</h3>
              <p className="text-sm text-red-800 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
