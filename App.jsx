import { useEffect, useMemo, useState } from 'react'

const API = import.meta.env.VITE_API_URL || ''

async function api(path, options = {}) {
  const token = localStorage.getItem('dermaai_token')
  const headers = new Headers(options.headers || {})
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(`${API}${path}`, { ...options, headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.error || 'Request failed')
  return data
}

function Auth({ onLogin }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setLoading(true)
    setMessage('')
    try {
      const data = await api(`/api/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (mode === 'login') {
        localStorage.setItem('dermaai_token', data.token)
        onLogin(data.username)
      } else {
        setMode('login')
        setMessage('Account created. You can now sign in.')
      }
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-card">
        <div className="brand-mark">D</div>
        <p className="eyebrow">DERMAAI</p>
        <h1>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
        <p className="muted">Educational skin-image classification. Not a medical diagnosis.</p>
        <form onSubmit={submit} className="stack">
          <label>Username<input value={username} onChange={(e) => setUsername(e.target.value)} required minLength={3} maxLength={64} /></label>
          <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={12} maxLength={128} /></label>
          <button disabled={loading}>{loading ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
        </form>
        {message && <p className="notice">{message}</p>}
        <button className="link-button" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setMessage('') }}>
          {mode === 'login' ? 'Need an account? Register' : 'Already have an account? Sign in'}
        </button>
      </section>
    </main>
  )
}

function Dashboard({ username }) {
  const [stats, setStats] = useState(null)
  const [recent, setRecent] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api('/api/dashboard/stats'), api('/api/dashboard/recent?limit=8')])
      .then(([statsData, recentData]) => { setStats(statsData); setRecent(recentData) })
      .catch((e) => setError(e.message))
  }, [])

  const cards = useMemo(() => stats ? [
    ['Predictions', stats.total], ['Normal', stats.normal], ['Psoriasis', stats.psoriasis], ['Ringworm', stats.ringworm], ['Acne', stats.acne], ['Patients', stats.patients_count],
  ] : [], [stats])

  return <section className="page">
    <div className="hero"><div><p className="eyebrow">DASHBOARD</p><h1>Hello, {username}</h1><p className="muted">Your private prediction history and summary.</p></div></div>
    {error && <p className="error">{error}</p>}
    <div className="stat-grid">{cards.map(([label, value]) => <div className="stat-card" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    <div className="panel"><h2>Recent predictions</h2>{recent.length === 0 ? <p className="muted">No predictions yet.</p> : <div className="table-wrap"><table><thead><tr><th>Result</th><th>Confidence</th><th>Patient</th><th>Date</th></tr></thead><tbody>{recent.map((item, index) => <tr key={`${item.timestamp}-${index}`}><td>{item.prediction}</td><td>{item.confidence}%{item.uncertain ? ' · uncertain' : ''}</td><td>{item.patient?.name || '—'}</td><td>{new Date(item.timestamp).toLocaleString()}</td></tr>)}</tbody></table></div>}</div>
  </section>
}

function Prediction() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [patient, setPatient] = useState({ name: '', age: '', phone: '' })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function chooseFile(event) {
    const selected = event.target.files?.[0]
    setFile(selected || null)
    setResult(null)
    if (selected) setPreview(URL.createObjectURL(selected))
  }

  async function submit(event) {
    event.preventDefault()
    if (!file) return setError('Choose an image first.')
    setLoading(true); setError(''); setResult(null)
    const body = new FormData()
    body.append('image', file)
    body.append('patientName', patient.name)
    body.append('patientAge', patient.age)
    body.append('patientPhone', patient.phone)
    try { setResult(await api('/api/predict', { method: 'POST', body })) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return <section className="page"><p className="eyebrow">ANALYSIS</p><h1>Skin image prediction</h1><p className="muted">Upload a clear skin image. Results are for educational use only.</p>
    <div className="two-col"><form className="panel stack" onSubmit={submit}><label>Image<input type="file" accept="image/*" onChange={chooseFile} required /></label>{preview && <img className="preview" src={preview} alt="Selected skin image" />}<div className="three-col"><label>Name<input value={patient.name} onChange={(e) => setPatient({ ...patient, name: e.target.value })} maxLength={100} /></label><label>Age<input type="number" min="0" max="120" value={patient.age} onChange={(e) => setPatient({ ...patient, age: e.target.value })} /></label><label>Phone<input value={patient.phone} onChange={(e) => setPatient({ ...patient, phone: e.target.value })} maxLength={30} /></label></div><button disabled={loading}>{loading ? 'Analyzing…' : 'Analyze image'}</button>{error && <p className="error">{error}</p>}</form>
      {result && <div className={`panel result ${result.uncertain ? 'warning' : ''}`}><p className="eyebrow">MODEL RESULT</p><h2>{result.prediction}</h2><div className="confidence">{result.confidence}%</div><p>{result.message}</p><h3>Probabilities</h3>{result.probabilities.map((p) => <div className="bar-row" key={p.label}><span>{p.label}</span><span>{p.probability}%</span><div className="bar"><i style={{ width: `${p.probability}%` }} /></div></div>)}<h3>General guidance</h3><ul>{result.tips.map((tip) => <li key={tip}>{tip}</li>)}</ul></div>}
    </div>
  </section>
}

function Hospitals() {
  const [city, setCity] = useState('')
  const [hospitals, setHospitals] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function search() {
    if (!city.trim()) return
    setLoading(true); setError('')
    try { const data = await api(`/api/hospitals/search-city?city=${encodeURIComponent(city.trim())}`); setHospitals(data.hospitals) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  function nearby() {
    if (!navigator.geolocation) return setError('Geolocation is not supported by this browser.')
    setLoading(true); setError('')
    navigator.geolocation.getCurrentPosition(async ({ coords }) => {
      try { setHospitals(await api(`/api/hospitals/nearby?lat=${coords.latitude}&lon=${coords.longitude}`)) }
      catch (e) { setError(e.message) }
      finally { setLoading(false) }
    }, () => { setError('Location permission was not granted.'); setLoading(false) })
  }

  return <section className="page"><p className="eyebrow">CARE</p><h1>Find a hospital</h1><p className="muted">Use these listings as a starting point and verify availability directly with the provider.</p><div className="search"><input placeholder="Search city" value={city} onChange={(e) => setCity(e.target.value)} /><button onClick={search} disabled={loading}>Search</button><button className="secondary" onClick={nearby} disabled={loading}>Use my location</button></div>{error && <p className="error">{error}</p>}<div className="hospital-grid">{hospitals.map((h) => <article className="hospital" key={`${h.name}-${h.city}`}><h3>{h.name}</h3><p>{h.specialization}</p><small>{h.address} · {h.city}</small>{h.distance != null && <strong>{h.distance} km away</strong>}<small>{h.timings}</small></article>)}</div></section>
}

function App() {
  const [username, setUsername] = useState(localStorage.getItem('dermaai_user') || '')
  const [view, setView] = useState('dashboard')

  function login(name) { localStorage.setItem('dermaai_user', name); setUsername(name); setView('dashboard') }
  function logout() { localStorage.removeItem('dermaai_token'); localStorage.removeItem('dermaai_user'); setUsername('') }

  if (!username) return <Auth onLogin={login} />
  return <div className="app-shell"><header><div className="brand"><span className="brand-mark small">D</span>DermaAI</div><nav>{[['dashboard','Dashboard'],['predict','Predict'],['hospitals','Hospitals']].map(([id, label]) => <button className={view === id ? 'nav-active' : ''} key={id} onClick={() => setView(id)}>{label}</button>)}</nav><button className="logout" onClick={logout}>Sign out</button></header>{view === 'dashboard' && <Dashboard username={username} />}{view === 'predict' && <Prediction />}{view === 'hospitals' && <Hospitals />}<footer>Educational/research prototype · Not a medical diagnosis</footer></div>
}

export default App
