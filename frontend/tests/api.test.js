import assert from 'node:assert/strict'
import test from 'node:test'
import { getPreview, getTrips, normalizePrediction, normalizeTrips, predict } from '../src/services/api.js'

const trip = { trip_id: '00000001', gps_points: [[12, 77]], predicted_route: [],
  predicted_arc_ids: [101, '00102'], model_name: 'Contract fixture', metrics: null }

test('accepts both catalogs, preserves leading zeros and rejects malformed IDs', () => {
  assert.deepEqual(normalizeTrips([trip.trip_id]), [trip.trip_id])
  assert.deepEqual(normalizeTrips({ trip_ids: [trip.trip_id, trip.trip_id] }), [trip.trip_id])
  assert.deepEqual(normalizeTrips([]), [])
  assert.throws(() => normalizeTrips([1]), /Invalid trip list/)
})

test('preserves arc types and handles empty geometry and unavailable metrics', () => {
  const result = normalizePrediction(trip, trip.trip_id)
  assert.deepEqual(result.segments, [])
  assert.deepEqual(result.metrics, {})
  assert.deepEqual(result.predicted_arc_ids, [101, '00102'])
  assert.deepEqual(normalizePrediction({ ...trip, metrics: undefined }, trip.trip_id).metrics, {})
})

test('normalizes flat geometry and keeps disconnected pieces separate', () => {
  const first = [[12, 77], [12.01, 77.01]]
  const second = [[13, 78], [13.01, 78.01]]
  assert.deepEqual(normalizePrediction({ ...trip, predicted_route: first }, trip.trip_id).segments, [first])
  assert.deepEqual(normalizePrediction({ ...trip, predicted_route: [first, second] }, trip.trip_id).segments, [first, second])
})

test('rejects malformed coordinates, unsafe numeric IDs, and stale trip IDs', () => {
  assert.throws(() => normalizePrediction({ ...trip, gps_points: [[91, 77]] }, trip.trip_id), /Invalid prediction/)
  assert.throws(() => normalizePrediction({ ...trip, predicted_route: [[12, '77']] }, trip.trip_id), /Invalid reconstructed/)
  assert.throws(() => normalizePrediction({ ...trip, predicted_arc_ids: [Number.MAX_SAFE_INTEGER + 1] }, trip.trip_id), /Invalid prediction/)
  assert.throws(() => normalizePrediction(trip, '00000002'), /Invalid prediction/)
})

test('all demo trips have GPS observations and synthetic route geometry without scores', async () => {
  const ids = await getTrips()
  assert.equal(ids.length, 3)
  await Promise.all(ids.map(async id => {
    const result = await predict(id)
    assert.deepEqual(result.gps_points, getPreview(id))
    assert.ok(result.gps_points.length && result.segments[0].length > 1)
    assert.match(result.model_name, /synthetic/i)
    assert.deepEqual(result.metrics, {})
  }))
})

test('aborted demo prediction does not return a result', async () => {
  const controller = new AbortController()
  const pending = predict('DEMO-01', controller.signal)
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
})
