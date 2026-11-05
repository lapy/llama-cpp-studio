/** Arrow, Home, and End keys for a WAI-ARIA tablist. Activates and focuses the next tab. */
export function onRovingTabKeydown(event) {
  const key = event.key
  if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(key)) return
  const tablist = event.currentTarget?.closest?.('[role="tablist"]') || event.currentTarget?.parentElement
  if (!tablist) return
  const tabs = [...tablist.querySelectorAll('[role="tab"]:not([disabled])')]
  const index = tabs.indexOf(event.currentTarget)
  if (index < 0 || !tabs.length) return
  let next = index
  if (key === 'ArrowRight' || key === 'ArrowDown') next = (index + 1) % tabs.length
  else if (key === 'ArrowLeft' || key === 'ArrowUp') next = (index - 1 + tabs.length) % tabs.length
  else if (key === 'Home') next = 0
  else if (key === 'End') next = tabs.length - 1
  event.preventDefault()
  const target = tabs[next]
  target.focus()
  target.click()
}
