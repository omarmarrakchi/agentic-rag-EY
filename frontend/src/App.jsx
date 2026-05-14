import { useState, useRef, useEffect } from 'react'
import './App.css'

const SESSION_ID = crypto.randomUUID()

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Bonjour ! Je suis votre assistant EY spécialisé dans l\'analyse des Termes de Référence. Posez-moi vos questions sur les marchés publics.',
      type: 'text',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

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
              if (next[assistantIndex].type !== 'tool') {
                next[assistantIndex] = { role: 'assistant', content: event.content, type: 'tool' }
              }
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

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <span className="ey-logo">EY</span>
          <div className="header-titles">
            <h1>Assistant TdR</h1>
            <p>Analyse de Termes de Référence — Marchés Publics</p>
          </div>
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg, i) => (
          <div key={i} className={`message-row ${msg.role}`}>
            {msg.role === 'assistant' && (
              <div className="avatar">EY</div>
            )}
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
        <button
          className="send-btn"
          onClick={sendMessage}
          disabled={loading || !input.trim()}
        >
          {loading ? '…' : 'Envoyer'}
        </button>
      </footer>
    </div>
  )
}
