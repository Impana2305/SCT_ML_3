import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

/* global tf, mobilenet */

const CAT_KW = ['cat','tabby','tiger cat','persian','siamese','egyptian','tom','wildcat','lynx','cougar','cheetah','leopard','lion','jaguar','kitten','feline']
const DOG_KW = ['dog','puppy','hound','spaniel','terrier','retriever','shepherd','bulldog','poodle','beagle','collie','husky','labrador','dalmatian','chihuahua','dachshund','boxer','pug','malamute','samoyed','corgi','whippet','mastiff','canine','weimaraner','basenji','pinscher','schnauzer','setter','pointer','bloodhound']

function findKw(cls, list) { const l = cls.toLowerCase(); return list.find(k => l.includes(k)) || null }

function classify(preds) {
  let cs = 0, ds = 0; const mc = [], md = []
  preds.forEach(({ className, probability }) => {
    const ck = findKw(className, CAT_KW); const dk = findKw(className, DOG_KW)
    if (ck) { cs += probability; mc.push({ className, probability, keyword: ck }) }
    if (dk) { ds += probability; md.push({ className, probability, keyword: dk }) }
  })
  const total = cs + ds
  if (total === 0) return { label:'Unknown', emoji:'❓', catPct:50, dogPct:50, confidence:0, mc, md, preds, unknown:true }
  const isCat = cs >= ds; const catPct = Math.round(cs / total * 100)
  return { label: isCat?'Cat':'Dog', emoji: isCat?'🐱':'🐶', catPct, dogPct: 100-catPct, confidence: Math.round(Math.max(cs,ds)*100), mc, md, preds, unknown:false, color: isCat?'cat':'dog' }
}

const METRICS = [
  { label: 'Training Accuracy', value: '99.69%' },
  { label: 'Testing Accuracy',  value: '75.00%' },
  { label: 'Variance Retained', value: '40.4%' },
  { label: 'PCA Components',    value: '50'  },
]
const STATS = [
  { label: 'Training Images', value: '10,000' },
  { label: 'SVM Accuracy',    value: '75.00%' },
  { label: 'Server Latency',  value: '0ms'  },
]

export default function App() {
  const [theme, setTheme]         = useState('light')
  const [model, setModel]         = useState(null)
  const [modelLoading, setML]     = useState(false)
  const [image, setImage]         = useState(null)
  const [imageFile, setFile]      = useState(null)
  const [classifying, setClf]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [dragging, setDrag]       = useState(false)
  const [activeTab, setTab]       = useState('classification')
  const imgRef      = useRef(null)
  const fileInputRef = useRef(null)

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light')

  const loadModel = useCallback(async () => {
    if (model || modelLoading) return model
    setML(true); setError(null)
    try {
      await window.tf.ready()
      const m = await window.mobilenet.load({ version: 2, alpha: 1.0 })
      setModel(m); return m
    } catch(e) { setError('Could not load AI model. Check your connection.'); return null }
    finally { setML(false) }
  }, [model, modelLoading])

  const handleFile = useCallback(file => {
    if (!file || !file.type.startsWith('image/')) return setError('Please upload a valid image.')
    if (file.size > 10*1024*1024) return setError('File too large (max 10 MB).')
    setError(null); setResult(null); setFile(file); setImage(URL.createObjectURL(file))
  }, [])

  const doClassify = useCallback(async () => {
    if (!imgRef.current) return
    setClf(true); setResult(null); setError(null)
    try {
      let m = model; if (!m) m = await loadModel(); if (!m) return
      const p = await m.classify(imgRef.current, 10)
      setResult(classify(p)); setTab('classification')
    } catch(e) { setError('Classification failed. Try another image.') }
    finally { setClf(false) }
  }, [model, loadModel])

  const reset = useCallback(() => {
    setImage(null); setFile(null); setResult(null); setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  useEffect(() => {
    if (!image || !imgRef.current) return
    const img = imgRef.current; const go = () => doClassify()
    if (img.complete) go(); else img.addEventListener('load', go, { once: true })
    return () => img.removeEventListener('load', go)
  }, [image]) // eslint-disable-line

  return (
    <div className="app" data-theme={theme}>

      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-logo">
          <span className="nav-logo-icon">🐾</span>
          Cat vs Dog Classifier
        </div>
        <button className="theme-btn" onClick={toggleTheme} aria-label="Toggle theme" title="Toggle light / dark">
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </nav>

      {/* Page header */}
      <div className="page-header">
        <div className="page-title">
          <span className="page-title-icon">🐾</span>
          Cat vs Dog Classifier
        </div>
        <p className="page-sub">Upload a photo to identify the animal and see how the model reached its decision.</p>
      </div>

      {/* Layout */}
      <div className="layout">

        {/* Sidebar */}
        <aside className="sidebar">

          {/* Upload / Preview card */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Image Input</span>
              {image && <button className="btn-sm btn-ghost" onClick={reset} style={{padding:'3px 10px'}}>Reset</button>}
            </div>

            {!image ? (
              <div
                className={`upload-zone${dragging?' dragging':''}`}
                onDrop={e=>{e.preventDefault();setDrag(false);handleFile(e.dataTransfer.files[0])}}
                onDragOver={e=>{e.preventDefault();setDrag(true)}}
                onDragLeave={()=>setDrag(false)}
                onClick={()=>fileInputRef.current?.click()}
                role="button" tabIndex={0} id="upload-zone"
                onKeyDown={e=>e.key==='Enter'&&fileInputRef.current?.click()}
              >
                <div className="upload-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <p className="upload-title">Drop image here</p>
                <p className="upload-hint">or click to browse · JPG PNG WebP</p>
              </div>
            ) : (
              <div className="sidebar-preview">
                <img ref={imgRef} src={image} alt="Uploaded" className="sidebar-img" crossOrigin="anonymous" />
                {(modelLoading||classifying) && (
                  <div className="sidebar-overlay">
                    <div className="spinner"/>
                    <span>{modelLoading?'Loading model...':'Analysing...'}</span>
                  </div>
                )}
                {!result&&!classifying&&!modelLoading && (
                  <div className="preview-actions">
                    <button className="btn-sm btn-primary" onClick={doClassify} id="classify-btn">Classify</button>
                    <button className="btn-sm btn-ghost" onClick={reset}>Reset</button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Stats card */}
          <div className="card stats-card">
            {STATS.map((s,i) => (
              <div className="stat-row" key={i}>
                <span className="stat-name">{s.label}</span>
                <span className="stat-val">{s.value}</span>
              </div>
            ))}
          </div>

          {error && <p className="error-bar">{error}</p>}
        </aside>

        {/* Right content */}
        <main className="content">
          <div className="tabs" role="tablist">
            {[['classification','Classification'],['metrics','Model Metrics']].map(([id,label])=>(
              <button key={id} className={`tab-btn${activeTab===id?' active':''}`} onClick={()=>setTab(id)} role="tab" aria-selected={activeTab===id}>{label}</button>
            ))}
          </div>

          {activeTab === 'classification' && (
            <div className="tab-content">
              {!result ? (
                <div className="empty-state">
                  <div className="empty-state-icon">🐾</div>
                  <p>Upload an image to see the classification result here.</p>
                </div>
              ) : (
                <ResultCard result={result}/>
              )}
            </div>
          )}

          {activeTab === 'metrics' && (
            <div className="tab-content">
              <div className="metrics-grid">
                {METRICS.map((m,i)=>(
                  <div className="metric-card card" key={i}>
                    <div className="metric-value">{m.value}</div>
                    <div className="metric-label">{m.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>

      </div>
      <input ref={fileInputRef} type="file" accept="image/*" onChange={e=>handleFile(e.target.files[0])} style={{display:'none'}} id="file-input"/>
    </div>
  )
}

function ResultCard({ result }) {
  const isCat = result.label === 'Cat'
  const matched = isCat ? result.mc : result.md
  const other   = isCat ? result.md : result.mc
  return (
    <div className="card" style={{animation:'fade-in 0.3s ease'}}>
      <div className={`verdict-card${result.color?' result-'+result.color:''}`}>
        <div className="verdict-emoji">{result.emoji}</div>
        <div>
          <div className="verdict-label-text">Classification Result</div>
          <div className={`verdict-animal ${result.unknown?'':result.color==='cat'?'verdict-cat':'verdict-dog'}`}>
            {result.unknown ? 'Unclear' : result.label}
          </div>
          {!result.unknown && <div className="verdict-conf">{result.confidence}% confidence · {result.catPct}% cat / {result.dogPct}% dog</div>}
        </div>
      </div>

      {!result.unknown && (
        <div className="bars-section">
          {[{label:'Cat',pct:result.catPct,type:'cat',active:isCat},{label:'Dog',pct:result.dogPct,type:'dog',active:!isCat}].map(b=>(
            <div className={`bar-row${b.active?' bar-active':''}`} key={b.label}>
              <span className="bar-label">{b.label}</span>
              <div className="bar-track"><div className={`bar-fill bar-${b.type}`} style={{'--target-width':`${b.pct}%`}}/></div>
              <span className="bar-pct">{b.pct}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="section-block-title">How it decided</div>
      {matched.length > 0 ? (
        <>
          <p className="reasoning-note">Detected labels indicating a <strong>{result.label?.toLowerCase()}</strong>:</p>
          <ul className="match-list">
            {matched.map((m,i)=>(
              <li className="match-item" key={i}>
                <span className="match-name">{m.className}</span>
                <span className="match-kw">"{m.keyword}"</span>
                <span className="match-pct">{(m.probability*100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </>
      ) : <p className="reasoning-note">No strong animal labels found in top predictions.</p>}

      {other.length > 0 && (
        <>
          <p className="reasoning-note" style={{paddingTop:0}}>Weaker signals for the other class:</p>
          <ul className="match-list dim">
            {other.map((m,i)=>(
              <li className="match-item" key={i}>
                <span className="match-name">{m.className}</span>
                <span className="match-kw">"{m.keyword}"</span>
                <span className="match-pct">{(m.probability*100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <details className="raw-wrap">
        <summary>All raw predictions</summary>
        <ul className="raw-list">
          {result.preds.map((p,i)=>(
            <li key={i}><span>{p.className}</span><span className="raw-pct">{(p.probability*100).toFixed(2)}%</span></li>
          ))}
        </ul>
      </details>
    </div>
  )
}

