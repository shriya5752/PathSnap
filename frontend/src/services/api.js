import { demoTrips } from '../fixtures/mockData.js'

const env = import.meta.env ?? {}
export const demoMode = (env.VITE_DEMO_MODE ?? 'true') === 'true'
const base = (env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

export function isCoordinate(point) {
  return Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)
    && Math.abs(point[0]) <= 90 && Math.abs(point[1]) <= 180
}

export function normalizeTrips(payload) {
  const ids = Array.isArray(payload) ? payload : payload?.trip_ids
  if (!Array.isArray(ids) || !ids.every(id => typeof id === 'string' && id.trim())) {
    throw new Error('Invalid trip list from the API.')
  }
  return [...new Set(ids)]
}

export function normalizePrediction(data, tripId) {
  if (!data || data.trip_id !== tripId || !Array.isArray(data.gps_points)
    || !data.gps_points.every(isCoordinate) || !Array.isArray(data.predicted_route)
    || !Array.isArray(data.predicted_arc_ids) || !data.predicted_arc_ids.every(id =>
      (typeof id === 'string' && id.length > 0) || Number.isSafeInteger(id))
    || typeof data.model_name !== 'string' || !data.model_name.trim()) {
    throw new Error('Invalid prediction: check the trip ID, coordinates, arc IDs and model name.')
  }
  const route = data.predicted_route
  const segments = !route.length ? [] : route.every(isCoordinate) ? [route] : route
  if (!segments.every(segment => Array.isArray(segment) && segment.every(isCoordinate))) {
    throw new Error('Invalid reconstructed route coordinates.')
  }
  const metrics = data.metrics ?? {}
  if (typeof metrics !== 'object' || Array.isArray(metrics)) throw new Error('Invalid prediction metrics.')
  return { ...data, segments, metrics }
}

async function request(path, { signal, ...options } = {}) {
  const timeout = AbortSignal.timeout(45000)
  let response
  let payload
  try {
    response = await fetch(`${base}${path}`, { ...options, signal: signal ? AbortSignal.any([signal, timeout]) : timeout })
    payload = await response.json().catch(error => {
      if (error instanceof SyntaxError) return undefined
      throw error
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new Error(timeout.aborted ? `${path} timed out.` : `Cannot reach ${base}${path}. Check the backend, network and CORS.`, { cause: error })
  }
  if (!response.ok) {
    const detail = typeof payload?.detail === 'string' ? `: ${payload.detail}` : ''
    throw new Error(`${path} returned HTTP ${response.status}${detail}`)
  }
  if (payload === undefined) throw new Error(`${path} did not return valid JSON.`)
  return payload
}

export async function getHealth(signal) {
  const health = await request('/health', { signal })
  if (health?.status !== 'ok') throw new Error('Unexpected health response.')
  return health
}

export async function getTrips(signal) {
  return demoMode ? Object.keys(demoTrips) : normalizeTrips(await request('/trips', { signal }))
}

export function getPreview(tripId) {
  return demoMode ? demoTrips[tripId]?.gps_points : undefined
}

export async function predict(tripId, signal) {
  if (demoMode) {
    await new Promise(resolve => setTimeout(resolve, 350))
    signal?.throwIfAborted()
    return normalizePrediction(demoTrips[tripId], tripId)
  }
  return normalizePrediction(await request('/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trip_id: tripId }), signal,
  }), tripId)
}
