import { defineComponent, nextTick, ref } from 'vue'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { useFocusReturn, watchDialogFocus } from './useFocusReturn'

describe('useFocusReturn', () => {
  it('returns focus to the control that opened the dialog', () => {
    const opener = document.createElement('button')
    opener.textContent = 'Connect'
    document.body.appendChild(opener)
    opener.focus()
    const { remember, restore } = useFocusReturn()
    remember()
    document.body.focus()
    restore()
    expect(document.activeElement).toBe(opener)
    opener.remove()
  })

  it('focuses the main landmark when the opener is gone', () => {
    const main = document.createElement('main')
    main.id = 'main-content'
    main.tabIndex = -1
    document.body.appendChild(main)
    const opener = document.createElement('button')
    document.body.appendChild(opener)
    opener.focus()
    const { remember, restore } = useFocusReturn()
    remember()
    opener.remove()
    restore()
    expect(document.activeElement).toBe(main)
    main.remove()
  })

  it('restores focus after a dialog closes', async () => {
    const opener = document.createElement('button')
    document.body.appendChild(opener)
    opener.focus()
    const visible = ref(false)
    const Harness = defineComponent({
      setup() {
        watchDialogFocus(visible)
        return () => null
      },
    })
    mount(Harness)
    visible.value = true
    await nextTick()
    document.body.focus()
    visible.value = false
    await nextTick()
    await nextTick()
    expect(document.activeElement).toBe(opener)
    opener.remove()
  })
})
