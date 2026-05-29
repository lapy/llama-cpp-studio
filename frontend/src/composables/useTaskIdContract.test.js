import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { cancelEndpointForTask } from './useTaskCancelEndpoint.js'
import { FORBIDDEN_PATTERNS, REAL_TASK_FIXTURES } from '@/test-fixtures/taskFixtures.js'

const FRONTEND_SRC = resolve(process.cwd(), 'frontend/src')

function walkJsVueFiles(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    const stat = statSync(path)
    if (stat.isDirectory()) {
      if (name === 'node_modules' || name === 'test-fixtures') continue
      walkJsVueFiles(path, out)
    } else if (/\.(js|vue)$/.test(name) && !/\.test\.js$/.test(name)) {
      out.push(path)
    }
  }
  return out
}

describe('task ID contract (frontend)', () => {
  it.each(REAL_TASK_FIXTURES)(
    'routes cancel for $label using real task_id',
    ({ task, cancelEndpoint }) => {
      expect(cancelEndpointForTask(task)).toBe(cancelEndpoint)
      expect(task.task_id).not.toMatch(/_operation$/)
    },
  )

  it('does not reference legacy synthetic task IDs or unified cancel API', () => {
    const files = walkJsVueFiles(FRONTEND_SRC)
    const violations = []

    for (const file of files) {
      const text = readFileSync(file, 'utf8')
      for (const pattern of FORBIDDEN_PATTERNS) {
        if (pattern.test(text)) {
          violations.push(`${relative(FRONTEND_SRC, file)}: ${pattern}`)
        }
      }
    }

    expect(violations).toEqual([])
  })

  it('stores SSE tasks keyed by task_id without rewriting IDs', () => {
    const progressSource = readFileSync(join(FRONTEND_SRC, 'stores/progress.js'), 'utf8')
    expect(progressSource).toContain('tasks.value = { ...tasks.value, [task.task_id]: task }')
    expect(progressSource).not.toMatch(/normalize\w+Task/)
    expect(progressSource).not.toMatch(/cuda_operation|lmdeploy_operation|onecat_vllm_operation/)
  })
})
