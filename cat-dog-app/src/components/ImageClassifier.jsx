import { useState, useRef, useCallback, useEffect } from 'react'
// tf and mobilenet are loaded from CDN as window globals (see index.html)
/* global tf, mobilenet */
import './ImageClassifier.css'

const CAT_KEYWORDS = [
  'cat','tabby','tiger cat','persian','siamese','egyptian','tom',
  'wildcat','lynx','cougar','cheetah','leopard','lion','jaguar',
  'snow leopard','kitten','feline','kit fox'
]
const DOG_KEYWORDS = [
  'dog','puppy','hound','spaniel','terrier','retriever','shepherd',
  'bulldog','poodle','beagle','collie','husky','labrador','dalmatian',
  'chihuahua','dachshund','boxer','pug','malamute','samoyed','corgi',
  'whippet','mastiff','canine','weimaraner','basenji','pinscher',
  'schnauzer','setter','pointer','bloodhound'
]

function matchKeyword(className, keywords) {
  const lower = className.toLowerCase()
  return keywords.find(k => lower.includes(k)) || null
}

function classifyPredictions(predictions) {
  let catScore = 0
  let dogScore = 0
  const matchedCat = []
  const matchedDog = []

  predictions.forEach(({ className, probability }) => {
    const catKw = matchKeyword(className, CAT_KEYWORDS)
    const dogKw = matchKeyword(className, DOG_KEYWORDS)
    if (catKw) {
      catScore += probability
      matchedCat.push({ className, probability, keyword: catKw })
    }
    if (dogKw) {
      dogScore += probability
      matchedDog.push({ className, probability, keyword: dogKw })
    }
  })

  const total = catScore + dogScore
  if (total === 0) {
    return {
      label: 'Unknown',
      emoji: '🤔',
      catPct: 50,
      dogPct: 50,
      confidence: 0,
      matchedCat,
      matchedDog,
      rawPredictions: predictions,
      unknown: true,
    }
  }

  const isCat = catScore >= dogScore
  const catPct = Math.round((catScore / total) * 100)
  const dogPct = 100 - catPct

  return {
    label: isCat ? 'Cat' : 'Dog',
    emoji: isCat ? '🐱' : '🐶',
    catPct,
    dogPct,
    confidence: Math.round(Math.max(catScore, dogScore) * 100),
    matchedCat,
    matchedDog,
    rawPredictions: predictions,
    unknown: false,
    color: isCat ? 'cat' : 'dog',
  }
}

function ImageClassifier() {
  const [model, setModel]             = useState(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [image, setImage]             = useState(null)
  const [imageFile, setImageFile]     = useState(null)
  const [classifying, setClassifying] = useState(false)
  const [result, setResult]           = useState(null)
  const [dragging, setDragging]       = useState(false)
  const [error, setError]             = useState(null)

  const imgRef      = useRef(null)
  const fileInputRef = useRef(null)

  const loadModel = useCallback(async () => {
    if (model || modelLoading) return model
    setModelLoading(true)
    setError(null)
    try {
      await window.tf.ready()
      const loaded = await window.mobilenet.load({ version: 2, alpha: 1.0 })
      setModel(loaded)
      return loaded
    } catch (err) {
      setError('Failed to load AI model. Please check your connection.')
      console.error(err)
      return null
    } finally {
      setModelLoading(false)
    }
  }, [model, modelLoading])

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload a valid image file.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File too large — please use an image under 10 MB.')
      return
    }
    setError(null)
    setResult(null)
    setImageFile(file)
    setImage(URL.createObjectURL(file))
  }, [])

  const handleDrop      = useCallback((e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }, [handleFile])
  const handleDragOver  = useCallback((e) => { e.preventDefault(); setDragging(true) }, [])
  const handleDragLeave = useCallback(() => setDragging(false), [])
  const handleFileInput = useCallback((e) => handleFile(e.target.files[0]), [handleFile])

  const handleClassify = useCallback(async () => {
    if (!imgRef.current) return
    setClassifying(true)
    setResult(null)
    setError(null)
    try {
      let activeModel = model
      if (!activeModel) activeModel = await loadModel()
      if (!activeModel) return
      const preds = await activeModel.classify(imgRef.current, 10)
      setResult(classifyPredictions(preds))
    } catch (err) {
      setError('Classification failed. Please try a different image.')
      console.error(err)
    } finally {
      setClassifying(false)
    }
  }, [model, loadModel])

  const handleReset = useCallback(() => {
    setImage(null); setImageFile(null); setResult(null); setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  useEffect(() => {
    if (!image || !imgRef.current) return
    const img = imgRef.current
    const go = () => handleClassify()
    if (img.complete) { go() } else { img.addEventListener('load', go, { once: true }) }
    return () => img.removeEventListener('load', go)
  }, [image]) // eslint-disable-line

  return (
    <div className="clf">
      {/* Drop Zone */}
      {!image && (
        <div
          className={`drop-zone ${dragging ? 'dragging' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          id="upload-zone"
          aria-label="Upload an image"
          onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
        >
          <div className="drop-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <p className="drop-title">Drop an image here</p>
          <p className="drop-hint">or click to browse · JPG, PNG, WebP</p>
        </div>
      )}

      {/* Preview + Result */}
      {image && (
        <div className="preview-wrap fade-in">
          <div className="preview-bar">
            <span className="preview-name">{imageFile?.name}</span>
            <button className="btn-reset" onClick={handleReset} id="reset-btn">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <polyline points="1 4 1 10 7 10"/>
                <path d="M3.51 15a9 9 0 102.13-9.36L1 10"/>
              </svg>
              Try another
            </button>
          </div>

          <div className="preview-img-wrap">
            <img ref={imgRef} src={image} alt="Uploaded" className="preview-img" crossOrigin="anonymous" />
            {(modelLoading || classifying) && (
              <div className="preview-overlay">
                <div className="spinner" />
                <span>{modelLoading ? 'Loading model…' : 'Analysing…'}</span>
              </div>
            )}
          </div>

          {result && <ResultPanel result={result} />}

          {!result && !classifying && !modelLoading && (
            <button className="btn-classify" onClick={handleClassify} id="classify-btn">
              Classify image
            </button>
          )}
        </div>
      )}

      {error && (
        <p className="error-msg" role="alert">{error}</p>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        style={{ display: 'none' }}
        id="file-input"
      />
    </div>
  )
}

/* ── Result Panel ── */
function ResultPanel({ result }) {
  const isCat  = result.label === 'Cat'
  const matched = isCat ? result.matchedCat : result.matchedDog
  const other   = isCat ? result.matchedDog : result.matchedCat

  return (
    <div className="result-panel fade-in">

      {/* Verdict */}
      <div className="verdict">
        <span className="verdict-emoji">{result.emoji}</span>
        <div>
          <p className="verdict-label">{result.unknown ? 'Unclear' : result.label}</p>
          {!result.unknown && (
            <p className="verdict-conf">
              {result.confidence}% confidence · {result.catPct}% cat / {result.dogPct}% dog
            </p>
          )}
        </div>
      </div>

      {/* Confidence bars */}
      {!result.unknown && (
        <div className="bars">
          <Bar label="Cat" pct={result.catPct} type="cat" active={isCat} />
          <Bar label="Dog" pct={result.dogPct} type="dog" active={!isCat} />
        </div>
      )}

      {/* How it decided */}
      <div className="reasoning">
        <p className="reasoning-title">How it decided</p>
        {matched.length > 0 ? (
          <>
            <p className="reasoning-desc">
              The model detected the following labels that indicate a <strong>{result.label.toLowerCase()}</strong>:
            </p>
            <ul className="match-list">
              {matched.map((m, i) => (
                <li key={i} className="match-item">
                  <span className="match-name">{m.className}</span>
                  <span className="match-kw">matched "{m.keyword}"</span>
                  <span className="match-pct">{(m.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="reasoning-desc">No strong animal-specific labels were detected in the top predictions.</p>
        )}

        {other.length > 0 && (
          <>
            <p className="reasoning-desc" style={{ marginTop: 12 }}>
              The opposite class also had signals, but weaker:
            </p>
            <ul className="match-list match-list-dim">
              {other.map((m, i) => (
                <li key={i} className="match-item">
                  <span className="match-name">{m.className}</span>
                  <span className="match-kw">matched "{m.keyword}"</span>
                  <span className="match-pct">{(m.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      {/* Raw predictions */}
      <details className="raw-wrap">
        <summary>All raw model predictions</summary>
        <ul className="raw-list">
          {result.rawPredictions.map((p, i) => (
            <li key={i}>
              <span>{p.className}</span>
              <span className="raw-pct">{(p.probability * 100).toFixed(2)}%</span>
            </li>
          ))}
        </ul>
      </details>

    </div>
  )
}

function Bar({ label, pct, type, active }) {
  return (
    <div className={`bar-row ${active ? 'bar-active' : ''}`}>
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className={`bar-fill bar-${type}`} style={{ '--target-width': `${pct}%` }} />
      </div>
      <span className="bar-pct">{pct}%</span>
    </div>
  )
}

export default ImageClassifier
