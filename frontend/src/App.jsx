import { useEffect, useRef, useState } from 'react'
import { ArrowUpRight, ChevronDown, Compass, RotateCw } from 'lucide-react'
import RouteMap from './components/RouteMap.jsx'
import { demoMode, getHealth, getPreview, getTrips, predict } from './services/api.js'
import './App.css'

const empty = []
const views = [['raw', 'Raw GPS'], ['route', 'Reconstructed'], ['overlay', 'Overlay']]
const displayPercent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%`
  : typeof value === 'string' && value.trim() ? value : 'N/A'

export default function App() {
  const [catalog, setCatalog] = useState({ ids: [], status: 'loading' })
  const [connection, setConnection] = useState('Checking API')
  const [attempt, setAttempt] = useState(0)
  const [selected, setSelected] = useState('')
  const [result, setResult] = useState(null)
  const [mode, setMode] = useState('raw')
  const [status, setStatus] = useState('Ready')
  const [error, setError] = useState('')
  const active = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    getTrips(controller.signal).then(ids => {
      if (controller.signal.aborted) return
      setCatalog({ ids, status: 'ready' })
      setSelected(ids[0] ?? '')
    }).catch(reason => {
      if (!controller.signal.aborted) setCatalog({ ids: [], status: 'error', error: reason.message })
    })
    if (!demoMode) getHealth(controller.signal).then(() => {
      if (!controller.signal.aborted) setConnection('API connected')
    }).catch(() => {
      if (!controller.signal.aborted) setConnection('API unavailable')
    })
    return () => controller.abort()
  }, [attempt])
  useEffect(() => () => active.current?.abort(), [])

  function selectTrip(tripId) {
    active.current?.abort()
    active.current = null
    setSelected(tripId)
    setResult(null)
    setMode('raw')
    setStatus('Ready')
    setError('')
  }

  async function reconstruct() {
    if (active.current || !selected) return
    const controller = new AbortController()
    active.current = controller
    setStatus('Reconstructing…')
    setResult(null)
    setMode('raw')
    setError('')
    try {
      const prediction = await predict(selected, controller.signal)
      if (controller.signal.aborted) return
      const hasGeometry = prediction.segments.some(segment => segment.length > 1)
      setResult(prediction)
      setMode(hasGeometry ? 'overlay' : 'raw')
      setStatus(hasGeometry ? 'Route available' : 'No route geometry')
      if (!demoMode) setConnection('API connected')
    } catch (reason) {
      if (controller.signal.aborted) return
      setError(reason.message)
      setStatus('Reconstruction failed')
    } finally {
      if (active.current === controller) active.current = null
    }
  }

  const gps = result?.gps_points ?? getPreview(selected)
  const segments = result?.segments ?? empty
  const hasRoute = segments.some(segment => segment.length > 1)
  const routeBlocks = result?.metrics.route_blocks
  const partialRoute = hasRoute && mode !== 'raw' && segments.length === 1 && Number.isInteger(routeBlocks) && routeBlocks > 1
  const busy = status === 'Reconstructing…'
  const syntheticBackend = !demoMode && /synthetic/i.test(result?.model_name ?? '')
  const metrics = [gps?.length ?? 'N/A', result ? new Set(result.predicted_arc_ids).size : 'N/A',
    displayPercent(result?.metrics.accuracy), displayPercent(result?.metrics.route_continuity)]
  const visibleStatus = catalog.status === 'loading' ? 'Loading trips…' : catalog.status === 'error'
    ? 'Trip list unavailable' : selected ? status : 'No trips available'
  const alert = error || catalog.error

  return <>
    <header className="masthead shell">
      <span className="wordmark">pathsnap<span>.</span></span>
      <div className="mast-meta"><span className="edition">THE ROUTE ATLAS / 2026</span>
        <span className="connection" role="status" data-online={connection === 'API connected'}>{demoMode ? 'Local fixtures' : connection}</span></div>
    </header>
    <main className="shell">
      <section className="intro" aria-labelledby="intro-title">
        <div><p className="eyebrow">01 / JOURNEY RECONSTRUCTION</p>
          <h1 id="intro-title">Finding the road<br />between the <em>points.</em></h1>
          <p className="subtitle">From noisy GPS observations to the road actually travelled.</p></div>
        <div className="atlas-index" aria-hidden="true"><Compass size={42} strokeWidth={0.8} /><span>ATLAS INDEX</span><span>FIG. 01 / ROUTE STUDY</span></div>
      </section>
      <section className="controls" aria-label="Journey controls">
        <label className="field"><span>SELECT JOURNEY</span><span className="select-wrap">
          <select value={selected} disabled={!catalog.ids.length} onChange={event => selectTrip(event.target.value)}>
            {!catalog.ids.length && <option value="">{catalog.status === 'loading' ? 'Loading…' : 'No journeys'}</option>}
            {catalog.ids.map(id => <option key={id}>{id}</option>)}
          </select><ChevronDown size={15} aria-hidden="true" /></span></label>
        <div className="journey-meta"><span className={`mode-tag ${demoMode || syntheticBackend ? 'demo' : ''}`}>{demoMode ? 'DEMO MODE' : syntheticBackend ? 'SYNTHETIC API RESULT' : 'LIVE API'}</span>
          <span role="status">{visibleStatus}</span></div>
        {catalog.status === 'error' ? <button className="action" onClick={() => {
          setCatalog({ ids: [], status: 'loading' })
          setConnection('Checking API')
          setAttempt(previous => previous + 1)
        }}>Retry connection <RotateCw size={16} aria-hidden="true" /></button> :
          <button className="action" disabled={!selected || busy} onClick={reconstruct}>{busy ? 'Reconstructing…' : 'Reconstruct Route'} <ArrowUpRight size={17} aria-hidden="true" /></button>}
      </section>
      {alert && <p className="alert" role="alert">{alert}</p>}
      <section className="plate" aria-labelledby="plate-title">
        <div className="plate-heading"><div><p className="eyebrow">02 / INTERACTIVE MAP</p><h2 id="plate-title">Journey plate</h2></div>
          <div className="segments" role="group" aria-label="Map layers">{views.map(([value, label]) =>
            <button key={value} disabled={value !== 'raw' && !hasRoute} aria-pressed={mode === value} onClick={() => setMode(value)}>{label}</button>)}</div>
        </div>
        <RouteMap key={selected} gps={gps ?? empty} segments={segments} mode={mode}
          emptyMessage={busy ? 'Waiting for reconstruction…' : result ? 'No GPS observations returned.' : selected ? 'Reconstruct this journey to load its GPS observations.' : 'No journey selected.'} />
        <div className="map-caption"><ul aria-label="Map legend"><li><i className="gps-symbol" />Raw GPS observations</li><li><i className="route-symbol" />Reconstructed route</li></ul>
          <span>{demoMode ? 'Synthetic geometry · not a road-network result' : syntheticBackend ? 'Synthetic backend fixture · not real inference' : partialRoute ? `Showing the longest of ${routeBlocks} decoded blocks` : hasRoute ? 'Geometry supplied by the API' : 'Reconstructed layers require route geometry'}</span></div>
      </section>
      <dl className="metrics" aria-label="Reconstruction metrics">
        {['GPS observations', 'Matched road segments', 'Accuracy', 'Route continuity'].map((label, index) =>
          <div key={label}><dt>0{index + 1} / {label}</dt><dd>{metrics[index]}</dd></div>)}
      </dl>
      <details className="details"><summary>View reconstruction details</summary>
        <dl><dt>Trip ID</dt><dd><code>{selected || 'N/A'}</code></dd>
          <dt>Model</dt><dd>{result?.model_name ?? 'N/A'}</dd>
          <dt>Status</dt><dd>{visibleStatus}</dd>
          <dt>Predicted arc IDs</dt><dd><code>{result ? JSON.stringify(result.predicted_arc_ids) : 'N/A'}</code></dd>
          <dt>API metrics</dt><dd><code>{result && Object.keys(result.metrics).length ? JSON.stringify(result.metrics) : 'N/A'}</code></dd>
          {alert && <><dt>Error</dt><dd>{alert}</dd></>}</dl>
      </details>
    </main>
    <footer className="shell footer"><strong>PATHSNAP</strong><span>GPS MAP MATCHING &amp; ROUTE RECONSTRUCTION</span><span>ACADEMIC PROJECT</span></footer>
  </>
}
