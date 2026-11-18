import { useEffect, useMemo } from 'react'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { isCoordinate } from '../services/api.js'

const center = [12.975, 77.595]

function MapViewport({ points }) {
  const map = useMap()
  useEffect(() => {
    if (points.length) map.fitBounds(points, { padding: [40, 40], maxZoom: 16, animate: false })
    else map.setView(center, 12, { animate: false })
  }, [map, points])
  useEffect(() => {
    const observer = new ResizeObserver(() => map.invalidateSize({ pan: false }))
    observer.observe(map.getContainer())
    return () => observer.disconnect()
  }, [map])
  return null
}

export default function RouteMap({ gps, segments, mode, emptyMessage }) {
  const observations = useMemo(() => mode === 'route' ? [] : gps.filter(isCoordinate), [gps, mode])
  const routes = useMemo(() => mode === 'raw' ? [] : segments.filter(segment =>
    Array.isArray(segment) && segment.length >= 2 && segment.every(isCoordinate)), [segments, mode])
  const bounds = useMemo(() => [...observations, ...routes.flat()], [observations, routes])
  return <div className="map-frame" role="region" aria-label="Interactive journey map">
    <MapContainer center={center} zoom={12} scrollWheelZoom={false}>
      <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19}
        attribution={'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'} />
      <MapViewport points={bounds} />
      {routes.map((segment, index) => <Polyline key={index} positions={segment}
        pathOptions={{ color: 'var(--route)', weight: 5, opacity: 0.95 }} />)}
      {observations.map((point, index) => <CircleMarker key={index} center={point} radius={5}
        pathOptions={{ color: 'white', weight: 1.5, fillColor: 'var(--gps)', fillOpacity: 1 }}>
        <Popup><strong>Observation {index + 1}</strong><br />Lat {point[0].toFixed(6)}<br />Lng {point[1].toFixed(6)}</Popup>
      </CircleMarker>)}
    </MapContainer>
    {!bounds.length && <div className="map-note">{emptyMessage}</div>}
  </div>
}
