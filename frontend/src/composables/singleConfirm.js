let confirmationOpen = false

/**
 * Open at most one app-wide confirmation at a time.
 *
 * PrimeVue's confirmation service is event based. Repeated UI events can publish the
 * same request more than once before the dialog becomes visible, which makes the
 * confirmation appear to reopen after it is dismissed. Keep the guard outside any
 * individual component because the app intentionally has a single ConfirmDialog host.
 */
export function requireSingleConfirmation(confirm, options) {
  if (confirmationOpen) return false

  confirmationOpen = true
  const release = () => {
    confirmationOpen = false
  }

  try {
    confirm.require({
      ...options,
      accept: async (...args) => {
        release()
        return options.accept?.(...args)
      },
      reject: (...args) => {
        release()
        return options.reject?.(...args)
      },
      onHide: (...args) => {
        release()
        return options.onHide?.(...args)
      },
    })
    return true
  } catch (error) {
    release()
    throw error
  }
}

