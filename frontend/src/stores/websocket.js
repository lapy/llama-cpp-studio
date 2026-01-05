import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useWebSocketStore = defineStore('websocket', () => {
  
  const ws = ref(null)
  const connected = ref(false)
  const reconnecting = ref(false)
  const lastMessage = ref(null)
  const messageHistory = ref([])
  const subscribers = ref(new Map())
  
  // Connection settings
  const maxReconnectAttempts = 10
  const reconnectDelay = 1000 // Start with 1 second
  const maxReconnectDelay = 30000 // Max 30 seconds
  let reconnectAttempts = 0
  let reconnectTimeout = null
  
  // Computed
  const connectionStatus = computed(() => {
    if (connected.value) return 'connected'
    if (reconnecting.value) return 'reconnecting'
    return 'disconnected'
  })
  
  const isConnected = computed(() => connected.value && ws.value?.readyState === WebSocket.OPEN)
  
  // WebSocket URL
  const getWebSocketUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${protocol}//${host}/ws`
  }
  
  // Connect to WebSocket
        const connect = () => {
            if (ws.value?.readyState === WebSocket.OPEN || ws.value?.readyState === WebSocket.CONNECTING) {
              return
            }
            
            const url = getWebSocketUrl()
            
            try {
              ws.value = new WebSocket(url)
              
        ws.value.onopen = () => {
          console.log('WebSocket connected successfully to:', url)
          connected.value = true
                reconnecting.value = false
                reconnectAttempts = 0
                
                // Send a test message to verify connection
                send({
                  type: 'test',
                  message: 'WebSocket connection test',
                  timestamp: new Date().toISOString()
                })
                
                // Send any queued messages
                flushQueuedMessages()
              }
              
        ws.value.onmessage = async (event) => {
          try {
            const data = JSON.parse(event.data)
            console.log('WebSocket received:', data)
            lastMessage.value = data
            messageHistory.value.push({
              ...data,
              timestamp: new Date().toISOString()
            })
            
            // Keep only last 100 messages
            if (messageHistory.value.length > 100) {
              messageHistory.value = messageHistory.value.slice(-100)
            }
            
            // Notify subscribers (now async)
            await notifySubscribers(data.type, data)
            
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }
              
              ws.value.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason)
                connected.value = false
                ws.value = null
                
                // Attempt reconnection if not manually closed
                if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
                  scheduleReconnect()
                }
              }
              
              ws.value.onerror = (error) => {
                console.error('WebSocket error:', error)
                connected.value = false
              }
              
            } catch (error) {
              console.error('Failed to create WebSocket connection:', error)
              scheduleReconnect()
            }
        }
  
  // Schedule reconnection with exponential backoff
  const scheduleReconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
    }
    
    reconnectAttempts++
    reconnecting.value = true
    
    const delay = Math.min(
      reconnectDelay * Math.pow(2, reconnectAttempts - 1),
      maxReconnectDelay
    )
    
    console.log(`Scheduling WebSocket reconnection in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`)
    
    reconnectTimeout = setTimeout(() => {
      connect()
    }, delay)
  }
  
  // Disconnect WebSocket
  const disconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
    
    reconnectAttempts = maxReconnectAttempts // Prevent reconnection
    
    if (ws.value) {
      ws.value.close(1000, 'Manual disconnect')
      ws.value = null
    }
    
    connected.value = false
    reconnecting.value = false
  }
  
  // Send message
  const send = (message) => {
    if (isConnected.value) {
      try {
        ws.value.send(JSON.stringify(message))
        return true
      } catch (error) {
        console.error('Failed to send WebSocket message:', error)
        return false
      }
    } else {
      console.warn('WebSocket not connected, message queued')
      queueMessage(message)
      return false
    }
  }
  
  // Message queue for when disconnected
  const queuedMessages = ref([])
  
  const queueMessage = (message) => {
    queuedMessages.value.push({
      ...message,
      timestamp: Date.now()
    })
  }
  
  const flushQueuedMessages = () => {
    while (queuedMessages.value.length > 0 && isConnected.value) {
      const message = queuedMessages.value.shift()
      send(message)
    }
  }
  
  // Subscribe to specific message types
  const subscribe = (messageType, callback) => {
    if (!subscribers.value.has(messageType)) {
      subscribers.value.set(messageType, new Set())
    }
    subscribers.value.get(messageType).add(callback)
    
    // Return unsubscribe function
    return () => {
      const callbacks = subscribers.value.get(messageType)
      if (callbacks) {
        callbacks.delete(callback)
        if (callbacks.size === 0) {
          subscribers.value.delete(messageType)
        }
      }
    }
  }
  
  // Notify subscribers
  const notifySubscribers = async (messageType, data) => {
    const callbacks = subscribers.value.get(messageType)
    if (callbacks) {
      // Use Promise.allSettled to handle both sync and async callbacks
      const promises = Array.from(callbacks).map(async (callback) => {
        try {
          const result = callback(data)
          // If callback returns a promise, await it
          if (result && typeof result.then === 'function') {
            await result
          }
        } catch (error) {
          console.error('Error in WebSocket subscriber callback:', error)
        }
      })
      
      // Wait for all callbacks to complete (or fail)
      await Promise.allSettled(promises)
    }
  }
  
  // Convenience methods for specific message types
  const subscribeToDownloadProgress = (callback) => {
    return subscribe('download_progress', callback)
  }
  
  const subscribeToBuildProgress = (callback) => {
    return subscribe('build_progress', callback)
  }
  
  const subscribeToModelStatus = (callback) => {
    return subscribe('model_status', callback)
  }
  
  const subscribeToSystemMetrics = (callback) => {
    return subscribe('system_metrics', callback)
  }
  
  const subscribeToNotifications = (callback) => {
    return subscribe('notification', callback)
  }
  
  const subscribeToRamUpdates = (callback) => {
    return subscribe('ram_update', callback)
  }
  
  const subscribeToVramUpdates = (callback) => {
    return subscribe('vram_update', callback)
  }
  
  const subscribeToDownloadComplete = (callback) => {
    return subscribe('download_complete', callback)
  }
  
  // Unified monitoring subscription for real-time system data
  const subscribeToUnifiedMonitoring = (callback) => {
    return subscribe('unified_monitoring', callback)
  }
  
  // Model events subscription for instant start/stop/loading updates
  const subscribeToModelEvents = (callback) => {
    return subscribe('model_event', callback)
  }
  
  // Get recent messages of specific type
  const getRecentMessages = (messageType, limit = 10) => {
    return messageHistory.value
      .filter(msg => msg.type === messageType)
      .slice(-limit)
  }
  
  // Clear message history
  const clearHistory = () => {
    messageHistory.value = []
  }
  
  return {
    // State
    connected,
    reconnecting,
    lastMessage,
    messageHistory,
    
    // Computed
    connectionStatus,
    isConnected,
    
    // Methods
    connect,
    disconnect,
    send,
    subscribe,
    subscribeToDownloadProgress,
    subscribeToBuildProgress,
    subscribeToModelStatus,
    subscribeToSystemMetrics,
    subscribeToNotifications,
    subscribeToRamUpdates,
    subscribeToVramUpdates,
    subscribeToDownloadComplete,
    subscribeToUnifiedMonitoring,
    subscribeToModelEvents,
    getRecentMessages,
    clearHistory
  }
})
