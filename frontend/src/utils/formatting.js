/**
 * Format bytes to human-readable string
 * @param {number} bytes - Bytes to format
 * @returns {string} Formatted string (e.g., "1.5 GB")
 */
export const formatBytes = (bytes) => {
  if (Number.isNaN(bytes) || bytes === null || bytes === undefined) return 'Unknown size'
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / Math.pow(1024, exponent)
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`
}

/**
 * Format file size (alias for formatBytes)
 * @param {number} bytes - Bytes to format
 * @returns {string} Formatted string
 */
export const formatFileSize = formatBytes

/**
 * Format date to locale string
 * @param {string|Date} dateString - Date string or Date object
 * @returns {string} Formatted date string
 */
export const formatDate = (dateString) => {
  if (!dateString) return 'Unknown'
  if (dateString instanceof Date) {
    return dateString.toLocaleString()
  }
  return new Date(dateString).toLocaleString()
}

/**
 * Format number with locale string
 * @param {number} num - Number to format
 * @returns {string} Formatted number string
 */
export const formatNumber = (num) => {
  if (num === null || num === undefined || Number.isNaN(num)) return '0'
  return num.toLocaleString()
}

