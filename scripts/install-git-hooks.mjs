import { execSync } from 'node:child_process'
import { existsSync, lstatSync, mkdirSync, readlinkSync, symlinkSync, unlinkSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
let hooksDir
try {
  hooksDir = execSync('git rev-parse --git-path hooks', {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim()
} catch {
  process.exit(0)
}
if (!hooksDir) process.exit(0)

const absoluteHooks = resolve(root, hooksDir)
mkdirSync(absoluteHooks, { recursive: true })
const destination = join(absoluteHooks, 'pre-commit')
const source = join(root, 'scripts', 'git-hooks', 'pre-commit')
const link = relative(absoluteHooks, source)

if (existsSync(destination)) {
  const stat = lstatSync(destination)
  if (stat.isSymbolicLink() && readlinkSync(destination) === link) process.exit(0)
  if (!stat.isSymbolicLink()) {
    console.error('pre-commit: an existing hook is not managed by this repo, so it was left unchanged')
    process.exit(0)
  }
  unlinkSync(destination)
}
symlinkSync(link, destination)
