'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  extractHighlightPhrases,
  applyHighlightsToTextLayer,
  type HighlightPhrase,
} from '@/lib/highlightEngine'

// Configure pdf.js worker in the same module as Document/Page (required by react-pdf)
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Summary {
  id: number
  model_type: string
  summary_text: string
}

interface PdfViewerProps {
  documentId: string
  summaries: Summary[]
  highlightedModels: string[]
}

export default function PdfViewer({ documentId, summaries, highlightedModels }: PdfViewerProps) {
  const [numPages, setNumPages] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [scale, setScale] = useState(1.2)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const phrasesRef = useRef<HighlightPhrase[]>([])

  // Pre-compute phrases when summaries or highlighted models change
  useEffect(() => {
    phrasesRef.current = extractHighlightPhrases(summaries, highlightedModels)
  }, [summaries, highlightedModels])

  const applyHighlights = useCallback(() => {
    if (!containerRef.current) return
    // react-pdf uses this class for the text layer container
    const textLayer = containerRef.current.querySelector('.react-pdf__Page__textContent')
    if (!textLayer) return
    applyHighlightsToTextLayer(textLayer as HTMLElement, phrasesRef.current)
  }, [])

  // Re-apply highlights when toggled models change
  useEffect(() => {
    // Small delay to ensure DOM is ready after react-pdf renders
    const timer = setTimeout(applyHighlights, 100)
    return () => clearTimeout(timer)
  }, [highlightedModels, applyHighlights])

  const handleTextLayerRendered = useCallback(() => {
    // Delay slightly — text layer spans may not be fully populated yet
    setTimeout(applyHighlights, 150)
  }, [applyHighlights])

  const fileUrl = `${API_URL}/api/v1/documents/${documentId}/file`

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b bg-muted/30 px-3 py-1.5">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            title="Previous page"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="min-w-[60px] text-center text-xs text-muted-foreground">
            {currentPage} / {numPages}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
            disabled={currentPage >= numPages}
            title="Next page"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale((s) => Math.max(0.5, +(s - 0.1).toFixed(1)))}
            title="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </Button>
          <span className="min-w-[40px] text-center text-xs text-muted-foreground">
            {(scale * 100).toFixed(0)}%
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale((s) => Math.min(2.5, +(s + 0.1).toFixed(1)))}
            title="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* PDF Render Area */}
      <div ref={containerRef} className="flex-1 overflow-auto bg-muted/10 p-4">
        <div className="mx-auto" style={{ width: 'fit-content' }}>
          {error ? (
            <div className="flex flex-col items-center justify-center py-20">
              <p className="text-sm text-destructive">Failed to load PDF</p>
              <p className="mt-1 text-xs text-muted-foreground">The file may be missing or corrupted.</p>
            </div>
          ) : (
            <Document
              file={fileUrl}
              onLoadSuccess={({ numPages: n }) => {
                setNumPages(n)
                setLoading(false)
              }}
              onLoadError={() => {
                setError(true)
                setLoading(false)
              }}
              loading={
                <div className="flex flex-col items-center gap-2 py-20">
                  <Skeleton className="h-[800px] w-[600px]" />
                </div>
              }
            >
              <Page
                pageNumber={currentPage}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={true}
                onRenderTextLayerSuccess={handleTextLayerRendered}
                loading={<Skeleton className="h-[800px] w-[600px]" />}
              />
            </Document>
          )}
        </div>
      </div>

      {/* Footer info */}
      {!loading && !error && (
        <div className="border-t bg-muted/30 px-3 py-1">
          <p className="text-[10px] text-muted-foreground">
            {numPages} page{numPages !== 1 ? 's' : ''} &middot; Use zoom controls to resize &middot; Toggle model highlights above
          </p>
        </div>
      )}
    </div>
  )
}
