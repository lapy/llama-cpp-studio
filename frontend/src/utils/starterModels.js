/**
 * Curated first-run model suggestions.
 *
 * These are Hugging Face repository ids, not catalog entries: Search resolves
 * them through the normal download flow, so this module names no engine and
 * promises no download API. Keep the list small, popular, and GGUF-only so a
 * first download fits the default setup.
 */
export const STARTER_MODELS = Object.freeze([
  {
    repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    name: 'Qwen2.5 0.5B Instruct',
    why: 'The fastest first download. Small enough to run on CPU and confirm the whole pipeline works.',
    size: 'About 0.5 GB at Q4_K_M',
  },
  {
    repo: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    name: 'Qwen2.5 1.5B Instruct',
    why: 'Noticeably better answers while still fitting a laptop. A good everyday local chat model.',
    size: 'About 1.1 GB at Q4_K_M',
  },
  {
    repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
    name: 'Llama 3.2 1B Instruct',
    why: 'A different model family with widely used quantizations, handy for trying Llama chat templates.',
    size: 'About 0.8 GB at Q4_K_M',
  },
])

/** Search deep link for one repository (the Search view owns `?q=`). */
export function starterModelSearchPath(repoId) {
  return `/search?q=${encodeURIComponent(repoId)}`
}
