/**
 * Curated first-run model suggestions.
 *
 * These are Hugging Face repository ids, not catalog entries: Search resolves
 * them through the normal download flow, so this module names no engine and
 * promises no download API. Keep the list small, popular, and GGUF-only so a
 * first download fits the default setup.
 *
 * Verified 2026-11-15 against the Hugging Face API with queries such as
 * https://huggingface.co/api/models?search=LFM2.5&filter=gguf&sort=downloads&direction=-1
 * plus each repo's `?blobs=true` file listing. Exact Q4_K_M files:
 *   LiquidAI/LFM2.5-350M-GGUF: LFM2.5-350M-Q4_K_M.gguf (0.21 GiB)
 *   LiquidAI/LFM2.5-1.2B-Instruct-GGUF: LFM2.5-1.2B-Instruct-Q4_K_M.gguf (0.68 GiB)
 *   ibm-granite/granite-4.2-3b-GGUF: granite-4.2-3b-Q4_K_M.gguf (2.09 GiB)
 *
 * Qwen3.5-4B was considered for the quality step-up, but its GGUF repos are
 * multimodal and most Q4_K_M builds exceed 2.6 GiB. Granite 4.2 3B is a
 * text-only instruction model and stays under that cap.
 */
export const STARTER_MODELS = Object.freeze([
  {
    repo: 'LiquidAI/LFM2.5-350M-GGUF',
    name: 'LFM2.5 350M',
    why: 'The smallest option here. A current instruction-tuned model that runs on CPU and proves the download-to-endpoint path.',
    size: 'About 0.2 GB at Q4_K_M',
  },
  {
    repo: 'LiquidAI/LFM2.5-1.2B-Instruct-GGUF',
    name: 'LFM2.5 1.2B Instruct',
    why: 'A current 1.2B text-only instruction model for laptops and edge devices, with fast short-form chat.',
    size: 'About 0.7 GB at Q4_K_M',
  },
  {
    repo: 'ibm-granite/granite-4.2-3b-GGUF',
    name: 'Granite 4.2 3B',
    why: 'A current text-only reasoning model when you have a little more RAM; a quality step up for local chat.',
    size: 'About 2.1 GB at Q4_K_M',
  },
])

/** Search deep link for one repository (the Search view owns `?q=`). */
export function starterModelSearchPath(repoId) {
  return `/search?q=${encodeURIComponent(repoId)}`
}
