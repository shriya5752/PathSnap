const shape = [[12.9769,77.5902],[12.9755,77.5911],[12.9746,77.5927],[12.9741,77.5948],[12.9739,77.5974],[12.9746,77.5992],[12.9752,77.6011]]

export const demoTrips = Object.fromEntries([0, 0.012, -0.019].map((offset, index) => {
  const tripId = `DEMO-0${index + 1}`
  const route = shape.map(([lat, lng]) => [lat + offset, lng + offset])
  return [tripId, {
    trip_id: tripId,
    gps_points: route.map(([lat, lng], step) => [lat + (step % 2 ? -0.0003 : 0.0004), lng + 0.0002]),
    predicted_route: route,
    predicted_arc_ids: route.slice(1).map((_, step) => `SYN-${index + 1}-${step + 1}`),
    metrics: null,
    model_name: 'Synthetic frontend fixture — no inference',
  }]
}))
