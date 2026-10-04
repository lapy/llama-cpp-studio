import { describe, expect, it } from 'vitest'
import { connectEndpoint, connectRequest, inferenceOrigin, normalizePublicInferenceUrl } from './connectTarget'

describe('connect targets', () => {
  it('keeps a local proxy on HTTP when the management page is HTTPS', () => {
    expect(inferenceOrigin({ proxyPort: 2345 })).toBe('http://127.0.0.1:2345')
  })

  it('uses a configured public URL instead of the local default', () => {
    expect(inferenceOrigin({
      publicUrl: 'https://infer.example.test/studio/',
      proxyPort: 2000,
    })).toBe('https://infer.example.test/studio')
  })

  it('rejects credentials and non-http schemes', () => {
    expect(normalizePublicInferenceUrl('https://user:secret@infer.example.test')).toBe('')
    expect(normalizePublicInferenceUrl('javascript:alert(1)')).toBe('')
  })

  it('builds a chat example and an embeddings example', () => {
    const chat = connectRequest({ id: 'org/model', llama_swap_id: 'org-model' })
    expect(chat.kind).toBe('chat')
    expect(chat.path).toBe('/v1/chat/completions')
    expect(chat.body.max_tokens).toBe(16)
    expect(connectEndpoint({ llama_swap_id: 'org-model' }, { proxyPort: 2000 }))
      .toBe('http://127.0.0.1:2000/v1/chat/completions')

    const embedding = connectRequest({
      id: 'org/embed',
      is_embedding_model: true,
      llama_swap_id: 'org-embed',
    })
    expect(embedding.kind).toBe('embeddings')
    expect(embedding.path).toBe('/v1/embeddings')
    expect(embedding.body.input).toBe('ping')
    expect(connectEndpoint(
      { is_embedding_model: true, llama_swap_id: 'org-embed' },
      { publicUrl: 'https://infer.example.test' },
    )).toBe('https://infer.example.test/v1/embeddings')
  })
})
