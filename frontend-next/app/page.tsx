"use client";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { getOrCreateUserId, resetIdentity } from "./lib/identity";

type Source = { document_name: string; score: number; section_title?: string | null; preview?: string };
type ReasoningStep = { stage: string; reasoning: string; done: boolean; startedAt?: number };
type Citation = { claim: string; source_label: string; document_name: string; proof_text: string; section_title?: string | null };
type Message = { role: "user" | "bot"; text: string; sources?: Source[]; reasoning?: ReasoningStep[]; citations?: Citation[]; question?: string; feedback?: 0 | 1; tab?: "answer" | "sources" | "thinking"; qualityWarning?: string; usageStats?: { promptTokens: number; completionTokens: number; elapsedSec: number }[] };
type DocInfo = { id: number; document_name: string; document_type: string | null; source_type: string | null; topic_tags: string[] | null; total_pages: number | null };
type Session = { id: string; title: string; updated_at: string };
type Stats = { documents: number; chunks: number };

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const STYLE_OPTIONS = ["summarized", "normal", "deep"];
const STYLE_LABELS = ["Summarized", "Normal", "Deep Explained"];
const STYLE_HINTS = ["3-5 lines", "5-10 lines", "10-20 lines, detailed"];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "ready" | "error">("checking");
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [docSearch, setDocSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("All");
  const [stats, setStats] = useState<Stats | null>(null);
  const [lengthIdx, setLengthIdx] = useState(1); // index into STYLE_OPTIONS
  const [deepMode, setDeepMode] = useState(false);
  const [agentMode, setAgentMode] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>("");
  const [showIdentity, setShowIdentity] = useState(false);
  const [latestSources, setLatestSources] = useState<Source[]>([]);
  const [activeCitation, setActiveCitation] = useState<{ msgIndex: number; citation: Citation } | null>(null);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { getOrCreateUserId().then(setUserId); }, []);

  function handleResetIdentity() {
    resetIdentity();
    getOrCreateUserId().then(setUserId);
    setShowIdentity(false);
  }

  async function loadSessions() {
    if (!userId) return;
    const r = await fetch(`${API_BASE}/api/sessions?user_id=${encodeURIComponent(userId)}`);
    const d = await r.json();
    setSessions(d.sessions || []);
  }

  async function newSession() {
    const r = await fetch(`${API_BASE}/api/sessions?user_id=${encodeURIComponent(userId)}`, { method: "POST" });
    const s = await r.json();
    setActiveSessionId(s.id);
    setMessages([]);
    setLatestSources([]);
    loadSessions();
  }

  async function switchSession(id: string) {
    setActiveSessionId(id);
    const r = await fetch(`${API_BASE}/api/sessions/${id}/messages`);
    const d = await r.json();
    setMessages(
      (d.messages || []).map((m: any) => ({
        role: m.role,
        text: m.content,
        sources: m.sources || undefined,
      }))
    );
  }

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => (r.ok ? setApiStatus("ready") : setApiStatus("error")))
      .catch(() => setApiStatus("error"));

    fetch(`${API_BASE}/api/documents`)
      .then((r) => r.json())
      .then((d) => setDocs(d.documents || []))
      .catch(() => {});

    fetch(`${API_BASE}/api/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (userId) loadSessions();
  }, [userId]);

  const availableTypes = Array.from(new Set(docs.map((d) => d.document_type || "other")));
  const filteredDocs = docs.filter((d) => {
    const matchesSearch = d.document_name.toLowerCase().includes(docSearch.toLowerCase());
    const matchesType = typeFilter === "All" || (d.document_type || "other") === typeFilter;
    return matchesSearch && matchesType;
  });

  async function rateMessage(index: number, rating: 0 | 1) {
    const msg = messages[index];
    if (!msg.question) return;
    setMessages((m) => m.map((x, i) => (i === index ? { ...x, feedback: rating } : x)));
    try {
      await fetch(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: msg.question, answer: msg.text, rating }),
      });
    } catch {}
  }

  function setMessageTab(index: number, tab: "answer" | "sources" | "thinking") {
    setMessages((m) => m.map((x, i) => (i === index ? { ...x, tab } : x)));
  }

  function renderAnswerWithCitations(text: string, citations: Citation[] | undefined, msgIndex: number) {
    if (!citations || citations.length === 0) {
      return <ReactMarkdown>{text}</ReactMarkdown>;
    }
    const parts = text.split(/(\[\d+\])/g);
    return (
      <p style={{ margin: 0 }}>
        {parts.map((part, i) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (!match) return <span key={i}>{part}</span>;
          const label = match[1];
          const citation = citations.find((c) => c.source_label === label);
          if (!citation) return null;
          return (
            <button
              key={i}
              className="citation-marker"
              onClick={() => setActiveCitation({ msgIndex, citation })}
              title="Click to see exact source text"
            >
              {label}
            </button>
          );
        })}
      </p>
    );
  }

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    let sessionId = activeSessionId;
    if (!sessionId) {
      const r = await fetch(`${API_BASE}/api/sessions?user_id=${encodeURIComponent(userId)}`, { method: "POST" });
      const s = await r.json();
      sessionId = s.id;
      setActiveSessionId(s.id);
    }
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);
    setMessages((m) => [...m, { role: "bot", text: "", question: q }]);
    const botIndex = messages.length + 1;

    try {
      const res = await fetch(`${API_BASE}/api/query-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, answer_style: STYLE_OPTIONS[lengthIdx], session_id: sessionId, user_id: userId, deep_mode: deepMode, agent_mode: agentMode }),
      });
      if (!res.body) throw new Error("No response stream");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = JSON.parse(line.slice(6));

          if (payload.type === "stage_start") {
            setMessages((m) => m.map((x, i) => (i === botIndex
              ? { ...x, reasoning: [...(x.reasoning || []), { stage: payload.stage, reasoning: "", done: false, startedAt: Date.now() }], tab: "thinking" }
              : x)));
          } else if (payload.type === "thinking_delta") {
            setMessages((m) => m.map((x, i) => {
              if (i !== botIndex) return x;
              const steps = [...(x.reasoning || [])];
              const idx = steps.findIndex((s) => s.stage === payload.stage && !s.done);
              if (idx >= 0) steps[idx] = { ...steps[idx], reasoning: steps[idx].reasoning + payload.text };
              return { ...x, reasoning: steps };
            }));
          } else if (payload.type === "stage_done") {
            setMessages((m) => m.map((x, i) => {
              if (i !== botIndex) return x;
              const steps = [...(x.reasoning || [])];
              const idx = steps.findIndex((s) => s.stage === payload.stage && !s.done);
              if (idx >= 0) steps[idx] = { stage: payload.stage, reasoning: payload.summary, done: true };
              return { ...x, reasoning: steps };
            }));
          } else if (payload.type === "sources") {
            setMessages((m) => m.map((x, i) => (i === botIndex ? { ...x, sources: payload.sources } : x)));
            setLatestSources(payload.sources || []);
          } else if (payload.type === "token") {
            setMessages((m) => m.map((x, i) => (i === botIndex ? { ...x, text: x.text + payload.text, tab: "answer" } : x)));
          } else if (payload.type === "usage_stats") {
            setMessages((m) => m.map((x, i) => (i === botIndex
              ? { ...x, usageStats: [...(x.usageStats || []), { promptTokens: payload.prompt_tokens, completionTokens: payload.completion_tokens, elapsedSec: payload.elapsed_sec }] }
              : x)));
          } else if (payload.type === "citations") {
            setMessages((m) => m.map((x, i) => (i === botIndex ? { ...x, citations: payload.citations } : x)));
          } else if (payload.type === "quality_warning") {
            setMessages((m) => m.map((x, i) => (i === botIndex ? { ...x, qualityWarning: payload.reasoning } : x)));
          } else if (payload.type === "done") {
            loadSessions();
          }
        }
      }
    } catch (e: any) {
      setMessages((m) => m.map((x, i) => (i === botIndex ? { ...x, text: "Error: " + e.message } : x)));
    }
    setLoading(false);
  }

  return (
    <div className="app-shell">
      {/* LEFT SIDEBAR */}
      {leftOpen ? (
        <div className="sidebar">
          <div className="sidebar-header">
            <div className="brand-mini">
              <img src="/iota-logo.png" alt="IOTA" className="brand-logo-img" />
              <span className="brand-mini-name">SAMA NORA</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className={`live-dot ${apiStatus !== "ready" ? "down" : ""}`} title={apiStatus} />
              <button className="icon-btn" onClick={() => setLeftOpen(false)}>«</button>
            </div>
          </div>

          <div className="sidebar-scroll">
            <div className="info-card">
              <div className="info-eyebrow">IOTA Local</div>
              <div className="info-title">Regulation AI</div>
              <div className="info-desc">Fully local AI answers with citations from your ingested SAMA/NCA document corpus.</div>
              <div className="pillrow">
                <span className={`pill ${apiStatus === "ready" ? "pill-green" : ""}`}>
                  {apiStatus === "ready" && <span className="pill-dot" />}
                  {apiStatus === "ready" ? "Live · healthy" : apiStatus === "error" ? "API unreachable" : "checking..."}
                </span>
                <span className="pill">Qwen3-4B</span>
                <span className="pill">BGE-M3</span>
                <span className="pill">Hybrid RAG</span>
                <span className="pill pill-orange">BGE-Reranker</span>
              </div>
              <div className="info-stats">
                <div>
                  <div className="info-stat-label">Docs ingested</div>
                  <div className="info-stat-value">{stats?.documents ?? "—"}</div>
                </div>
                <div>
                  <div className="info-stat-label">Chunks</div>
                  <div className="info-stat-value">{stats?.chunks ?? "—"}</div>
                </div>
              </div>
            </div>

            <div className="new-chat-wrap">
              <button className="new-chat-btn" onClick={newSession}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" /></svg>
                New chat
              </button>
            </div>

            <div className="section-label">Chats</div>
            <div className="thread-list">
              {sessions.length === 0 && <div className="empty" style={{ padding: "4px 10px" }}>No past chats yet.</div>}
              {sessions.map((s) => (
                <button
                  key={s.id}
                  className={`thread-item ${s.id === activeSessionId ? "active" : ""}`}
                  onClick={() => switchSession(s.id)}
                >
                  <span className="thread-title">{s.title || "New chat"}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-footer">
            <button className="profile-btn" onClick={() => setShowIdentity((v) => !v)}>
              <span className="profile-avatar">{userId ? userId[0]?.toUpperCase() : "N"}</span>
              <span style={{ minWidth: 0 }}>
                <div className="profile-name">Local user</div>
                <div className="profile-id mono">{userId ? userId.slice(0, 10) + "..." : "..."}</div>
              </span>
            </button>
            {showIdentity && (
              <div style={{
                position: "absolute", bottom: 60, left: 12, background: "#fff", border: "1px solid oklch(0.9 0.006 264)",
                borderRadius: 12, padding: 14, boxShadow: "0 20px 40px oklch(0.2 0.05 264 / 0.15)", zIndex: 10, minWidth: 220,
              }}>
                <div style={{ fontSize: 11, color: "oklch(0.5 0.14 150)", fontWeight: 600, marginBottom: 6 }}>● Recognized via Fingerprint</div>
                <div style={{ fontSize: 11, color: "oklch(0.55 0.01 264)", marginBottom: 4 }}>YOUR DEVICE ID</div>
                <div className="mono" style={{ fontSize: 12, wordBreak: "break-all" }}>{userId}</div>
                <div style={{ fontSize: 11, color: "oklch(0.65 0.01 264)", marginTop: 8 }}>
                  Your browser fingerprint identifies your device — no login required.
                </div>
                <button
                  onClick={handleResetIdentity}
                  style={{ marginTop: 10, width: "100%", border: "1px solid oklch(0.85 0.1 25)", background: "oklch(0.97 0.02 25)", color: "oklch(0.5 0.15 25)", borderRadius: 8, padding: "6px 0", fontSize: 12, cursor: "pointer" }}
                >
                  Reset identity
                </button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="sidebar collapsed">
          <img src="/iota-logo.png" alt="IOTA" className="collapsed-logo" />
          <button className="icon-btn-ghost" onClick={() => setLeftOpen(true)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
          </button>
          <button className="collapsed-new-btn" onClick={newSession}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" /></svg>
          </button>
        </div>
      )}

      {/* CENTER */}
      <div className="main-col">
        <div className="chat-header">
          <div>
            <div className="chat-header-title">
              {sessions.find((s) => s.id === activeSessionId)?.title || "New conversation"}
            </div>
            <div className="chat-header-sub">SAMA / NCA · {stats?.documents ?? 0} documents indexed</div>
          </div>
          <div className="offline-badge">
            <span className="offline-dot" />
            <span className="offline-text">Online</span>
          </div>
        </div>

        <div className="messages">
          {messages.length === 0 && <div className="empty">Ask a question about the ingested documents.</div>}
          {messages.map((m, i) => {
            const activeTab = m.tab || "answer";
            if (m.role === "user") {
              return (
                <div key={i} className="msg-user-row">
                  <div className="msg-user-bubble"><ReactMarkdown>{m.text}</ReactMarkdown></div>
                </div>
              );
            }
            const isEmpty = m.text === "";
            return (
              <div key={i} className="msg-bot-row">
                <div className="msg-bot-avatar"><span className="msg-bot-avatar-dot" /></div>
                <div className="msg-bot-body">
                  <div className="msg-bot-card">
                    <div className="msg-tabs">
                      <button className={`msg-tab ${activeTab === "answer" ? "active" : ""}`} onClick={() => setMessageTab(i, "answer")}>Answer</button>
                      {m.sources && m.sources.length > 0 && (
                        <button className={`msg-tab ${activeTab === "sources" ? "active" : ""}`} onClick={() => setMessageTab(i, "sources")}>
                          Sources ({m.sources.length})
                        </button>
                      )}
                      {m.reasoning && m.reasoning.length > 0 && (
                        <button className={`msg-tab ${activeTab === "thinking" ? "active" : ""}`} onClick={() => setMessageTab(i, "thinking")}>
                          Thinking ({m.reasoning.length})
                        </button>
                      )}
                    </div>
                    <div className="msg-tabs-divider" />
                    {activeTab === "answer" ? (
                      isEmpty ? (
                        <div className="typing-inline"><span className="dot" /><span className="dot" /><span className="dot" /></div>
                      ) : (
                        <div className="msg-answer-text">
                          {m.qualityWarning && (
                            <div style={{ background: "oklch(0.97 0.04 55)", border: "1px solid oklch(0.85 0.08 55)", borderRadius: 8, padding: "8px 10px", fontSize: 12, color: "oklch(0.45 0.1 55)", marginBottom: 10 }}>
                              ⚠️ This answer may be incomplete — see Thinking tab for details.
                            </div>
                          )}
                          {renderAnswerWithCitations(m.text, m.citations, i)}
                        </div>
                      )
                    ) : activeTab === "thinking" ? (
                      <div className="msg-sources-list">
                        {m.reasoning!.map((r, j) => (
                          <div key={j} className="msg-source-card">
                            <div className="msg-source-doc" style={{ marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
                              {r.stage}
                              {!r.done && <span className="typing-inline" style={{ padding: 0 }}><span className="dot" /><span className="dot" /><span className="dot" /></span>}
                              {!r.done && r.startedAt && <ElapsedTicker startedAt={r.startedAt} />}
                            </div>
                            <div className="msg-source-preview" style={{ whiteSpace: "pre-wrap" }}>{r.reasoning || "..."}</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="msg-sources-list">
                        {m.sources!.map((s, j) => (
                          <div key={j} className="msg-source-card">
                            <div className="msg-source-top">
                              <span className="msg-source-doc">{s.document_name}</span>
                              <span className="msg-source-score mono">{s.score}</span>
                            </div>
                            {s.section_title && <div className="msg-source-section">{s.section_title}</div>}
                            {s.preview && <div className="msg-source-preview">{s.preview}...</div>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  {!isEmpty && m.usageStats && m.usageStats.length > 0 && (
                    <div style={{ fontSize: 10.5, color: "oklch(0.6 0.01 264)", marginTop: 4 }}>
                      {m.usageStats.reduce((s, u) => s + u.promptTokens, 0)} in / {m.usageStats.reduce((s, u) => s + u.completionTokens, 0)} out tokens
                      {" · "}{m.usageStats.reduce((s, u) => s + u.elapsedSec, 0).toFixed(1)}s
                      {m.usageStats.length > 1 && ` · ${m.usageStats.length} calls`}
                    </div>
                  )}
                  {!isEmpty && m.question && (
                    <div className="feedback-row">
                      <span className="feedback-label">Was this accurate?</span>
                      <button className={`feedback-btn up ${m.feedback === 1 ? "active" : ""}`} onClick={() => rateMessage(i, 1)} title="Helpful">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M7 22V10l5-8a2 2 0 0 1 2 2v5h6a2 2 0 0 1 2 2.4l-2 8A2 2 0 0 1 18 21H7z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /><path d="M7 10H3v12h4" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /></svg>
                      </button>
                      <button className={`feedback-btn down ${m.feedback === 0 ? "active" : ""}`} onClick={() => rateMessage(i, 0)} title="Not helpful" style={{ transform: "scaleY(-1)" }}>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M7 22V10l5-8a2 2 0 0 1 2 2v5h6a2 2 0 0 1 2 2.4l-2 8A2 2 0 0 1 18 21H7z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /><path d="M7 10H3v12h4" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /></svg>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <div className="length-row">
            <span className="length-label">Answer style</span>
            <input
              type="range" min={0} max={2} step={1} value={lengthIdx}
              onChange={(e) => setLengthIdx(Number(e.target.value))}
              className="length-slider"
            />
            <span className="length-value mono">{STYLE_LABELS[lengthIdx]} ({STYLE_HINTS[lengthIdx]})</span>
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 16, fontSize: 11.5, fontWeight: 600, color: "oklch(0.5 0.01 264)", cursor: "pointer" }}>
              <input type="checkbox" checked={deepMode} onChange={(e) => { setDeepMode(e.target.checked); if (e.target.checked) setAgentMode(false); }} />
              Deep mode (shows reasoning, slower)
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 16, fontSize: 11.5, fontWeight: 600, color: "oklch(0.5 0.01 264)", cursor: "pointer" }}>
              <input type="checkbox" checked={agentMode} onChange={(e) => { setAgentMode(e.target.checked); if (e.target.checked) setDeepMode(false); }} />
              Agent mode (can search again, slowest)
            </label>
          </div>
          <div className="composer-row">
            <textarea
              className="composer-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask about a regulation, filing requirement, or control..."
              rows={1}
            />
            <button className="send-btn" onClick={send} disabled={loading}>
              Send
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
          </div>
          <div className="composer-footnote">
            AI can make mistakes. Verify critical guidance against primary source documents. · All processing occurs on-device — no data leaves this machine.
          </div>
        </div>
      </div>

      {/* RIGHT SIDEBAR */}
      {rightOpen ? (
        <div className="right-sidebar">
          <div className="right-header">
            <span className="section-label" style={{ padding: 0 }}>Context spotlight</span>
            <button className="icon-btn-ghost" onClick={() => setRightOpen(false)}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
            </button>
          </div>

          <div className="spotlight-box">
            {latestSources.length === 0 ? (
              <div className="empty" style={{ padding: 12 }}>Ask a question to see cited passages.</div>
            ) : (
              latestSources.map((s, i) => (
                <div key={i} className="spotlight-item">
                  <div className="spotlight-top">
                    <span className="spotlight-doc">{s.document_name}</span>
                    <span className="spotlight-score mono">{s.score}</span>
                  </div>
                  {s.section_title && <div className="spotlight-section">{s.section_title}</div>}
                  {s.preview && <div className="spotlight-preview">{s.preview}...</div>}
                </div>
              ))
            )}
          </div>

          <div className="docs-section">
            <span className="section-label" style={{ padding: 0 }}>Documents ({docs.length})</span>
            <div className="doc-search-wrap">
              <span className="doc-search-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" /><path d="M21 21l-4.3-4.3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
              </span>
              <input className="doc-search-input" placeholder="Search documents..." value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
            <div className="filter-row">
              <button className={`filter-pill ${typeFilter === "All" ? "active" : ""}`} onClick={() => setTypeFilter("All")}>All</button>
              {availableTypes.map((t) => (
                <button key={t} className={`filter-pill ${typeFilter === t ? "active" : ""}`} onClick={() => setTypeFilter(t)}>{t}</button>
              ))}
            </div>
            <div className="doc-scroll">
              {filteredDocs.length === 0 && <div className="empty">No documents found.</div>}
              {filteredDocs.map((d) => (
                <div key={d.id} className="doc-row">
                  <div className="doc-icon mono">{(d.document_type || "DOC").slice(0, 3).toUpperCase()}</div>
                  <div className="doc-info">
                    <div className="doc-name">{d.document_name}</div>
                    <div className="doc-meta">{d.document_type || "unclassified"}{d.total_pages ? ` · ${d.total_pages} pages` : ""}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="right-sidebar collapsed">
          <button className="icon-btn-ghost" onClick={() => setRightOpen(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><rect x="3" y="4" width="18" height="16" rx="2" stroke="currentColor" strokeWidth="1.8" /><path d="M15 4v16" stroke="currentColor" strokeWidth="1.8" /></svg>
          </button>
        </div>
      )}
      {activeCitation && (
        <div className="citation-overlay" onClick={() => setActiveCitation(null)}>
          <div className="citation-popover" onClick={(e) => e.stopPropagation()}>
            <div className="citation-popover-header">
              <span>Source [{activeCitation.citation.source_label}]</span>
              <button onClick={() => setActiveCitation(null)} className="icon-btn-ghost">✕</button>
            </div>
            <div className="citation-doc-name">{activeCitation.citation.document_name}</div>
            {activeCitation.citation.section_title && (
              <div className="citation-section">{activeCitation.citation.section_title}</div>
            )}
            <div className="citation-claim-label">Claim made:</div>
            <div className="citation-claim">{activeCitation.citation.claim}</div>
            <div className="citation-claim-label">Exact source text:</div>
            <div className="citation-proof">{activeCitation.citation.proof_text}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function ElapsedTicker({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(id);
  }, [startedAt]);
  return <span style={{ fontSize: 10, color: "oklch(0.6 0.01 264)" }}>{elapsed}s</span>;
}
