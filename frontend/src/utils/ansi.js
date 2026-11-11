/**
 * Minimal, dependency-free ANSI/VT renderer for terminal output in the UI.
 *
 * Engines and installers emit SGR colour codes (and occasionally other control
 * sequences). Rendering those raw on the page is unreadable, so this module
 * converts a line into safe text segments that Vue can render as spans. It is
 * intentionally small: it understands colours, the common text attributes,
 * OSC title/clipboard sequences, and strips every other escape sequence.
 */

const ESC = '\u001b'

const FG_NAMES = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

// xterm's first 16 colours. Only used for inline extended-colour results.
const SYSTEM_COLORS = [
  '#000000',
  '#800000',
  '#008000',
  '#808000',
  '#000080',
  '#800080',
  '#008080',
  '#c0c0c0',
  '#808080',
  '#ff0000',
  '#00ff00',
  '#ffff00',
  '#0000ff',
  '#ff00ff',
  '#00ffff',
  '#ffffff',
]

function emptyState() {
  return { classes: new Set(), color: null, background: null }
}

function resetState(state) {
  state.classes.clear()
  state.color = null
  state.background = null
}

function applySgr(state, rawParams) {
  const params = rawParams === '' ? ['0'] : rawParams.split(';')
  for (let index = 0; index < params.length; index += 1) {
    // "38:2::r:g:b" / "38:5:n" colon form (ISO 8613-6). Normalise separators.
    const token = params[index].includes(':') ? params[index].split(':') : [params[index]]
    const code = Number.parseInt(token[0], 10)
    if (!Number.isFinite(code)) continue

    if (code === 0) {
      resetState(state)
    } else if (code === 1) {
      state.classes.add('ansi-bold')
    } else if (code === 2) {
      state.classes.add('ansi-dim')
    } else if (code === 3) {
      state.classes.add('ansi-italic')
    } else if (code === 4) {
      state.classes.add('ansi-underline')
    } else if (code === 7) {
      state.classes.add('ansi-reverse')
    } else if (code === 22) {
      state.classes.delete('ansi-bold')
      state.classes.delete('ansi-dim')
    } else if (code === 23) {
      state.classes.delete('ansi-italic')
    } else if (code === 24) {
      state.classes.delete('ansi-underline')
    } else if (code === 27) {
      state.classes.delete('ansi-reverse')
    } else if (code >= 30 && code <= 37) {
      state.color = `var(--ansi-fg-${FG_NAMES[code - 30]})`
    } else if (code === 39) {
      state.color = null
    } else if (code >= 40 && code <= 47) {
      state.background = `var(--ansi-bg-${FG_NAMES[code - 40]})`
    } else if (code === 49) {
      state.background = null
    } else if (code >= 90 && code <= 97) {
      state.color = `var(--ansi-fg-bright-${FG_NAMES[code - 90]})`
    } else if (code >= 100 && code <= 107) {
      state.background = `var(--ansi-bg-bright-${FG_NAMES[code - 100]})`
    } else if (code === 38 || code === 48) {
      let color = null
      let consumed = 0
      if (token.length > 2) {
        // Colon form: 38:5:n or 38:2::r:g:b.
        if (Number(token[1]) === 5 && token[2] != null) {
          color = ansi256ToCss(Number(token[2]))
        } else if (Number(token[1]) === 2) {
          const values = token.slice(2).filter((part) => part !== '')
          if (values.length >= 3) {
            color = rgbToCss(Number(values[0]), Number(values[1]), Number(values[2]))
          }
        }
        consumed = 0
      } else if (Number(params[index + 1]) === 5) {
        color = ansi256ToCss(Number(params[index + 2]))
        consumed = 2
      } else if (Number(params[index + 1]) === 2) {
        color = rgbToCss(
          Number(params[index + 2]),
          Number(params[index + 3]),
          Number(params[index + 4]),
        )
        consumed = 4
      }
      if (color) {
        if (code === 38) state.color = color
        else state.background = color
      }
      index += consumed
    }
  }
}

function rgbToCss(red, green, blue) {
  if (![red, green, blue].every((value) => Number.isFinite(value))) return null
  const clamp = (value) => Math.max(0, Math.min(255, Math.round(value)))
  return `rgb(${clamp(red)} ${clamp(green)} ${clamp(blue)})`
}

/** Translate an xterm-256 palette index into a CSS colour. */
export function ansi256ToCss(index) {
  if (!Number.isFinite(index) || index < 0 || index > 255) return null
  if (index < 16) return SYSTEM_COLORS[index]
  if (index >= 232) {
    const value = 8 + (index - 232) * 10
    return `rgb(${value} ${value} ${value})`
  }
  const offset = index - 16
  const steps = [0, 95, 135, 175, 215, 255]
  return `rgb(${steps[Math.floor(offset / 36)]} ${steps[Math.floor((offset % 36) / 6)]} ${steps[offset % 6]})`
}

function pushSegment(segments, text, state) {
  if (!text) return
  const segment = { text }
  if (state.classes.size) segment.className = [...state.classes].join(' ')
  if (state.color) segment.style = { ...(segment.style || {}), color: state.color }
  if (state.background)
    segment.style = { ...(segment.style || {}), backgroundColor: state.background }
  segments.push(segment)
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function highlightSegments(segments, query) {
  const needle = String(query || '').trim()
  if (!needle) return segments
  const pattern = new RegExp(escapeRegExp(needle), 'gi')
  const highlighted = []
  for (const segment of segments) {
    let lastIndex = 0
    let match
    pattern.lastIndex = 0
    while ((match = pattern.exec(segment.text))) {
      if (match.index > lastIndex) {
        highlighted.push({ ...segment, text: segment.text.slice(lastIndex, match.index) })
      }
      highlighted.push({
        ...segment,
        text: match[0],
        className: [segment.className, 'log-match'].filter(Boolean).join(' '),
      })
      lastIndex = match.index + match[0].length
      if (match[0].length === 0) pattern.lastIndex += 1
    }
    if (lastIndex < segment.text.length) {
      highlighted.push({ ...segment, text: segment.text.slice(lastIndex) })
    }
  }
  return highlighted
}

/**
 * Convert one line of terminal output into render segments.
 *
 * @param {string} text Raw output line (may contain ANSI escapes).
 * @param {string} [query] Optional search text to wrap in highlight segments.
 * @returns {Array<{text: string, className?: string, style?: Object}>}
 */
export function renderAnsi(text, query = '') {
  const source = String(text ?? '')
  const segments = []
  const state = emptyState()
  let index = 0

  while (index < source.length) {
    const escapeAt = source.indexOf(ESC, index)
    if (escapeAt === -1) {
      pushSegment(segments, source.slice(index), state)
      break
    }
    if (escapeAt > index) pushSegment(segments, source.slice(index, escapeAt), state)

    const kind = source[escapeAt + 1]
    if (kind === '[') {
      // CSI: parameter/intermediate bytes then one final byte in @-~.
      let cursor = escapeAt + 2
      while (cursor < source.length) {
        const code = source.charCodeAt(cursor)
        if (code >= 0x40 && code <= 0x7e) break
        cursor += 1
      }
      if (cursor >= source.length) break
      if (source[cursor] === 'm') {
        applySgr(state, source.slice(escapeAt + 2, cursor))
      }
      index = cursor + 1
    } else if (kind === ']') {
      // OSC: terminated by BEL or ST (ESC \).
      let cursor = escapeAt + 2
      while (cursor < source.length) {
        if (source[cursor] === '\u0007') break
        if (source[cursor] === ESC && source[cursor + 1] === '\\') break
        cursor += 1
      }
      index = source[cursor] === '\u0007' ? cursor + 1 : cursor + 2
    } else if (kind !== undefined && '()*+-./'.includes(kind)) {
      // Charset designation (ESC ( B etc.) is three bytes.
      index = escapeAt + 3
    } else if (kind === undefined) {
      break
    } else {
      index = escapeAt + 2
    }
  }

  return highlightSegments(segments, query)
}

/** Strip ANSI escapes and optional inline-progress segments. */
export function stripAnsi(text) {
  return renderAnsi(text)
    .map((segment) => segment.text)
    .join('')
}
