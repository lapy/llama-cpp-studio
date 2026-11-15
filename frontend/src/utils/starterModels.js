/**
 * Curated first-run model suggestions.
 *
 * These are Hugging Face repository ids, not catalog entries: Search resolves
 * them through the normal download flow, so this module names no engine and
 * promises no download API. Keep the list small, popular, and GGUF-only so a
 * first download fits the default setup.
 *
 * Verified 2026-11-15 against the Hugging Face API with queries such as
 * https://huggingface.co/api/models?search=Qwen3&filter=gguf&sort=downloads&direction=-1
 * plus each repo's `?blobs=true` file listing for the Q4_K_M size below.
 */
export const STARTER_MODELS = Object.freeze([
  {
    repo: 'bartowski/Qwen_Qwen3-0.6B-GGUF',
    name: 'Qwen3 0.6B',
    why: 'The smallest option here. A current Qwen model that runs comfortably on CPU and proves the download-to-endpoint path.',
    size: 'About 0.4 GB at Q4_K_M',
  },
  {
    repo: 'LiquidAI/LFM2.5-1.2B-Instruct-GGUF',
    name: 'LFM2.5 1.2B Instruct',
    why: 'A current 1.2B instruction model for laptops and edge devices, with fast short-form chat.',
    size: 'About 0.7 GB at Q4_K_M',
  },
  {
    repo: 'ggml-org/SmolLM3-3B-GGUF',
    name: 'SmolLM3 3B',
    why: 'A multilingual 3B instruction model when you have a little more RAM; a quality step up for local chat.',
    size: 'About 1.8 GB at Q4_K_M',
  },
])

/** Search deep link for one repository (the Search view owns `?q=`). */
export function starterModelSearchPath(repoId) {
  return `/search?q=${encodeURIComponent(repoId)}`
}
