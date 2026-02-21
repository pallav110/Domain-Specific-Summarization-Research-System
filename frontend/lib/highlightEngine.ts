/**
 * N-gram based highlight engine for matching summary text to PDF source text.
 * Searches the concatenated text layer for n-gram matches, then highlights
 * the corresponding spans by setting backgroundColor (no DOM restructuring).
 */

export interface HighlightPhrase {
  text: string
  modelType: string
  color: string
}

export const MODEL_HIGHLIGHT_COLORS: Record<string, { bg: string; solid: string; label: string }> = {
  bart:                  { bg: 'rgba(249, 115, 22, 0.3)',  solid: '#f97316', label: 'BART' },
  pegasus:               { bg: 'rgba(139, 92, 246, 0.3)',  solid: '#8b5cf6', label: 'PEGASUS' },
  gemini:                { bg: 'rgba(59, 130, 246, 0.3)',   solid: '#3b82f6', label: 'Gemini' },
  gpt:                   { bg: 'rgba(16, 185, 129, 0.3)',  solid: '#10b981', label: 'GPT-4' },
  legal_bert_pegasus:    { bg: 'rgba(6, 182, 212, 0.3)',   solid: '#06b6d4', label: 'Legal-BERT' },
  clinical_bert_pegasus: { bg: 'rgba(236, 72, 153, 0.3)',  solid: '#ec4899', label: 'Clinical-BERT' },
}

function splitIntoSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10)
}

function extractNgrams(sentence: string, n: number): string[] {
  const words = sentence
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 0)

  if (words.length < n) return words.length > 0 ? [words.join(' ')] : []

  const ngrams: string[] = []
  for (let i = 0; i <= words.length - n; i++) {
    ngrams.push(words.slice(i, i + n).join(' '))
  }
  return ngrams
}

export function extractHighlightPhrases(
  summaries: Array<{ model_type: string; summary_text: string }>,
  enabledModels: string[]
): HighlightPhrase[] {
  const phrases: HighlightPhrase[] = []

  for (const summary of summaries) {
    if (!enabledModels.includes(summary.model_type)) continue

    const color = MODEL_HIGHLIGHT_COLORS[summary.model_type]?.bg || 'rgba(107, 114, 128, 0.3)'
    const sentences = splitIntoSentences(summary.summary_text)

    for (const sentence of sentences) {
      const ngrams = extractNgrams(sentence, 4)
      if (sentence.split(/\s+/).length <= 8) {
        ngrams.push(...extractNgrams(sentence, 3))
      }
      for (const ngram of ngrams) {
        phrases.push({ text: ngram, modelType: summary.model_type, color })
      }
    }
  }

  // Deduplicate by ngram text (keep first occurrence)
  const seen = new Set<string>()
  return phrases.filter((p) => {
    if (seen.has(p.text)) return false
    seen.add(p.text)
    return true
  })
}

/**
 * Apply highlights by searching concatenated text across all spans,
 * then coloring the spans that overlap with matches.
 * Only sets inline styles — no DOM restructuring — so pdf.js
 * absolute positioning is fully preserved.
 */
export function applyHighlightsToTextLayer(
  textLayerElement: HTMLElement,
  phrases: HighlightPhrase[]
): void {
  if (!textLayerElement) return

  // Collect all spans (use broad selector since structure varies by pdfjs version)
  const allSpans = Array.from(textLayerElement.querySelectorAll('span')) as HTMLSpanElement[]

  // Filter to only spans that contain actual text (skip empty/whitespace-only)
  const spans = allSpans.filter((s) => (s.textContent || '').trim().length > 0)

  // Clear all previous highlights
  for (const span of allSpans) {
    span.style.backgroundColor = ''
    span.style.borderRadius = ''
  }

  if (phrases.length === 0 || spans.length === 0) return

  // Build a map of span index -> { text, charStart } within the concatenated page text
  const spanMeta: Array<{ span: HTMLSpanElement; charStart: number; charEnd: number }> = []
  let fullText = ''
  for (const span of spans) {
    const text = span.textContent || ''
    const start = fullText.length
    fullText += text + ' '
    spanMeta.push({ span, charStart: start, charEnd: start + text.length })
  }

  const fullTextLower = fullText.toLowerCase()

  // Track which character ranges are already highlighted to avoid overlaps
  const highlightedRanges: Array<{ start: number; end: number; color: string }> = []

  // Search for each phrase in the concatenated text
  for (const phrase of phrases) {
    const phraseText = phrase.text.toLowerCase()
    let searchFrom = 0

    while (true) {
      const idx = fullTextLower.indexOf(phraseText, searchFrom)
      if (idx === -1) break
      searchFrom = idx + 1

      const matchEnd = idx + phraseText.length

      // Skip if overlapping with existing highlight
      const overlaps = highlightedRanges.some(
        (r) => idx < r.end && matchEnd > r.start
      )
      if (overlaps) continue

      highlightedRanges.push({ start: idx, end: matchEnd, color: phrase.color })
    }
  }

  // Now color the spans that overlap with any highlighted range
  for (const range of highlightedRanges) {
    for (const meta of spanMeta) {
      // Check if this span overlaps with the highlighted range
      if (meta.charEnd > range.start && meta.charStart < range.end) {
        // Skip tiny spans (page numbers, single chars) that might be metadata
        const trimmed = (meta.span.textContent || '').trim()
        if (trimmed.length < 3 && /^\d+$/.test(trimmed)) continue

        meta.span.style.backgroundColor = range.color
        meta.span.style.borderRadius = '2px'
      }
    }
  }
}
