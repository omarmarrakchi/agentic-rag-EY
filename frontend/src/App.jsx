import { useState, useRef, useEffect } from 'react'
import './App.css'

const SESSION_ID = crypto.randomUUID()

function scoreColor(score) {
  if (score >= 0.7) return '#16a34a'
  if (score >= 0.5) return '#d97706'
  return '#6b7280'
}

function truncate(text, max = 180) {
  if (!text) return ''
  return text.length <= max ? text : text.slice(0, max) + '…'
}

export default function App() {
  const [mode, setMode] = useState('chat')

  // ── Chat state ────────────────────────────────────────────────────────────
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: "Bonjour ! Je suis votre assistant EY spécialisé dans l'analyse des Termes de Référence. Posez-moi vos questions sur les marchés publics.",
    type: 'text',
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Search state ──────────────────────────────────────────────────────────
  const [query, setQuery] = useState('')
  const [filterOrg, setFilterOrg] = useState('')
  const [filterLieu, setFilterLieu] = useState('')
  const [results, setResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [searched, setSearched] = useState(false)
  const [selected, setSelected] = useState(null)

  // ── Chat handlers ─────────────────────────────────────────────────────────
  async function sendMessage() {
    const question = input.trim()
    if (!question || loading) return

    setInput('')
    setLoading(true)
    setMessages(prev => [...prev, { role: 'user', content: question }])

    let assistantIndex = null
    setMessages(prev => {
      assistantIndex = prev.length
      return [...prev, { role: 'assistant', content: '', type: 'text' }]
    })

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, session_id: SESSION_ID }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          let event
          try { event = JSON.parse(raw) } catch { continue }

          if (event.type === 'tool') {
            setMessages(prev => {
              const next = [...prev]
              if (next[assistantIndex].type !== 'tool')
                next[assistantIndex] = { role: 'assistant', content: event.content, type: 'tool' }
              return next
            })
          } else if (event.type === 'token') {
            setMessages(prev => {
              const next = [...prev]
              if (next[assistantIndex].type === 'tool') {
                next[assistantIndex] = { role: 'assistant', content: event.content, type: 'text' }
              } else {
                next[assistantIndex] = {
                  ...next[assistantIndex],
                  content: next[assistantIndex].content + event.content,
                  type: 'text',
                }
              }
              return next
            })
          } else if (event.type === 'error') {
            setMessages(prev => {
              const next = [...prev]
              next[assistantIndex] = { role: 'assistant', content: `Erreur : ${event.content}`, type: 'error' }
              return next
            })
          } else if (event.type === 'done') {
            break
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const next = [...prev]
        next[assistantIndex] = { role: 'assistant', content: `Erreur de connexion : ${err.message}`, type: 'error' }
        return next
      })
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // ── Search handlers ───────────────────────────────────────────────────────
  const canSearch = query.trim() || filterOrg.trim() || filterLieu.trim()

  async function doSearch(e) {
    e?.preventDefault()
    if (!canSearch || searching) return

    setSearching(true)
    setSearched(false)
    setSelected(null)
    setResults([])

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          organisation: filterOrg.trim(),
          lieu: filterLieu.trim(),
          k: 12,
        }),
      })
      const data = await res.json()
      setResults(data.results || [])
    } catch {
      setResults([])
    } finally {
      setSearching(false)
      setSearched(true)
    }
  }

  function openDetail(result, index) {
    setSelected({
      result,
      similar: results.filter((_, j) => j !== index).slice(0, 3),
    })
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <span className="ey-logo">EY</span>
          <div className="header-titles">
            <h1>Assistant TdR</h1>
            <p>Analyse de Termes de Référence — Marchés Publics</p>
          </div>
        </div>
        <nav className="mode-tabs">
          <button className={`mode-tab ${mode === 'chat' ? 'active' : ''}`} onClick={() => setMode('chat')}>
            Chat
          </button>
          <button className={`mode-tab ${mode === 'search' ? 'active' : ''}`} onClick={() => setMode('search')}>
            Recherche
          </button>
        </nav>
      </header>

      {/* ── Chat mode ────────────────────────────────────────────────────── */}
      {mode === 'chat' && (
        <>
          <main className="chat-window">
            {messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role}`}>
                {msg.role === 'assistant' && <div className="avatar">EY</div>}
                <div className={`bubble ${msg.role} ${msg.type === 'tool' ? 'tool' : ''} ${msg.type === 'error' ? 'error' : ''}`}>
                  {msg.type === 'tool' ? (
                    <span className="tool-indicator">
                      <span className="spinner" />
                      {msg.content}
                    </span>
                  ) : (
                    <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
                  )}
                  {msg.role === 'assistant' && msg.type === 'text' && loading && i === messages.length - 1 && msg.content === '' && (
                    <span className="typing-dots"><span /><span /><span /></span>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </main>

          <footer className="input-area">
            <textarea
              className="input-box"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Posez votre question sur les TdRs… (Entrée pour envoyer)"
              rows={1}
              disabled={loading}
            />
            <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
              {loading ? '…' : 'Envoyer'}
            </button>
          </footer>
        </>
      )}

      {/* ── Search mode ──────────────────────────────────────────────────── */}
      {mode === 'search' && (
        <div className="search-container">

          {/* Search bar + filters */}
          <form className="search-panel" onSubmit={doSearch}>
            <div className="search-bar-row">
              <input
                className="search-input"
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Mots-clés (optionnel si un filtre est rempli)"
                autoFocus
              />
              <button className="search-btn" type="submit" disabled={searching || !canSearch}>
                {searching ? '…' : 'Rechercher'}
              </button>
            </div>
            <div className="filters-row">
              <div className="filter-field">
                <label>Organisation / Bailleur</label>
                <input
                  type="text"
                  value={filterOrg}
                  onChange={e => setFilterOrg(e.target.value)}
                  placeholder="ex: PNUD, Banque Mondiale…"
                />
              </div>
              <div className="filter-field">
                <label>Lieu / Pays / Région</label>
                <input
                  type="text"
                  value={filterLieu}
                  onChange={e => setFilterLieu(e.target.value)}
                  placeholder="ex: Tunis, Maroc, Afrique…"
                />
              </div>
            </div>
          </form>

          {/* Results area */}
          <div className="results-area">
            {searching && (
              <div className="search-status">
                <span className="spinner" style={{ width: 20, height: 20, borderWidth: 3 }} />
                Recherche en cours…
              </div>
            )}

            {!searching && searched && results.length === 0 && (
              <div className="search-status">Aucun résultat trouvé pour cette recherche.</div>
            )}

            {!searching && results.length > 0 && (
              <>
                <p className="results-count">
                  {results.length} TdR{results.length > 1 ? 's' : ''} trouvé{results.length > 1 ? 's' : ''}
                </p>
                <div className="results-grid">
                  {results.map((r, i) => (
                    <div className="result-card" key={i}>
                      <div className="card-header">
                        <span className="score-badge" style={{ background: scoreColor(r.score) }}>
                          {Math.round(r.score * 100)}%
                        </span>
                        <span className="card-source" title={r.source}>{r.source}</span>
                      </div>
                      <h3 className="card-titre">{r.titre}</h3>
                      {r.organisation && <p className="card-org">{r.organisation}</p>}
                      <div className="card-pills">
                        {r.lieu        && <span className="pill pill-lieu">📍 {r.lieu}</span>}
                        {r.duree       && <span className="pill pill-duree">⏱ {r.duree}</span>}
                        {r.budget      && <span className="pill pill-budget">💰 {r.budget}</span>}
                        {r.date_limite && <span className="pill pill-date">📅 {r.date_limite}</span>}
                      </div>
                      {r.profil_consultant && (
                        <p className="card-profil">{truncate(r.profil_consultant)}</p>
                      )}
                      <button className="details-btn" onClick={() => openDetail(r, i)}>
                        Voir détails
                      </button>
                    </div>
                  ))}
                </div>
              </>
            )}

            {!searched && !searching && (
              <div className="search-placeholder">
                <p>Entrez des mots-clés pour rechercher dans la base des TdRs.</p>
                <p className="search-hint">
                  Ex : "expert formation entrepreneuriale Tunis" · "audit financier ONG" · "consultant infrastructure Afrique"
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Detail modal ─────────────────────────────────────────────────── */}
      {selected && (
        <div className="modal-overlay" onClick={() => setSelected(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelected(null)}>✕</button>

            <div className="modal-header">
              <span className="score-badge" style={{ background: scoreColor(selected.result.score) }}>
                {Math.round(selected.result.score * 100)}%
              </span>
              <span className="modal-source">{selected.result.source}</span>
            </div>

            <h2 className="modal-titre">{selected.result.titre}</h2>
            {selected.result.organisation && (
              <p className="modal-org">{selected.result.organisation}</p>
            )}

            <div className="card-pills" style={{ margin: '12px 0' }}>
              {selected.result.lieu        && <span className="pill pill-lieu">📍 {selected.result.lieu}</span>}
              {selected.result.duree       && <span className="pill pill-duree">⏱ {selected.result.duree}</span>}
              {selected.result.budget      && <span className="pill pill-budget">💰 {selected.result.budget}</span>}
              {selected.result.date_limite && <span className="pill pill-date">📅 {selected.result.date_limite}</span>}
            </div>

            {selected.result.profil_consultant && (
              <div className="modal-section">
                <h4>Profil du consultant</h4>
                <p>{selected.result.profil_consultant}</p>
              </div>
            )}

            {selected.result.objectifs?.length > 0 && (
              <div className="modal-section">
                <h4>Objectifs</h4>
                <ul>{selected.result.objectifs.map((o, i) => <li key={i}>{o}</li>)}</ul>
              </div>
            )}

            {selected.result.livrables?.length > 0 && (
              <div className="modal-section">
                <h4>Livrables attendus</h4>
                <ul>{selected.result.livrables.map((l, i) => <li key={i}>{l}</li>)}</ul>
              </div>
            )}

            {selected.similar?.length > 0 && (
              <div className="modal-section">
                <h4>Missions similaires</h4>
                <div className="similar-list">
                  {selected.similar.map((s, i) => (
                    <div key={i} className="similar-item" onClick={() => openDetail(s, results.indexOf(s))}>
                      <span className="score-badge score-badge-sm" style={{ background: scoreColor(s.score) }}>
                        {Math.round(s.score * 100)}%
                      </span>
                      <span className="similar-titre">{s.titre}</span>
                      {s.organisation && <span className="similar-org">{s.organisation}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
