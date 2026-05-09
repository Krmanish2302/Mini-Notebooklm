import { useState, useRef, useEffect } from "react";

// --- Constants ---
const MODES = [
  {id:"chat",  label:"Chat",     cls:""},
  {id:"deep",  label:"Research", cls:"amber"},
  {id:"study", label:"Study",    cls:"green"},
];
const EMB_MODELS = [
  {id:"all-MiniLM-L6-v2",  name:"MiniLM-L6 (Fast)",  dim:384},
  {id:"all-mpnet-base-v2",  name:"MPNet (Balanced)",  dim:768},
  {id:"e5-large-v2",        name:"E5-Large (Strong)", dim:1024},
];
const CHUNKERS = [
  {id:"recursive",   label:"Recursive"},
  {id:"page",        label:"Page-based"},
  {id:"paragraph",   label:"Paragraph-based"},
  {id:"sentence",    label:"Sentence-based"},
  {id:"semantic",    label:"Semantic"},
];
const SRC_ICONS = {pdf:"📄",youtube:"▶️",website:"🌐",text:"📝",image:"🖼️",csv:"📊",video:"🎬"};

// --- Components ---

function MiniGraph({ stats }) {
  const [hover, setHover] = useState(null);
  // Mock nodes based on real stats if available, else demo
  const nodes = [
    {x:50,y:38,hub:true, label:"Knowledge Core", chunks: stats?.total_chunks || 0, edges: stats?.graph?.edges || 0},
    {x:24,y:68,hub:false,label:"Vector Store",   chunks: stats?.chunks?.total_chunks || 0, edges: 0},
    {x:76,y:63,hub:false,label:"Entities",       chunks: stats?.graph?.nodes || 0, edges: 0},
  ];
  const edges = [[0,1],[0,2]];

  return (
    <div className="graph-canvas">
      <svg style={{position:"absolute",top:0,left:0,width:"100%",height:"100%"}}>
        {edges.map(([a,b],i)=>(
          <line key={i} x1={`${nodes[a].x}%`} y1={`${nodes[a].y}%`} x2={`${nodes[b].x}%`} y2={`${nodes[b].y}%`} stroke="var(--dim)" strokeWidth="1"/>
        ))}
      </svg>
      {nodes.map((n,i)=>(
        <div key={i} className={`graph-node${n.hub?" hub":""}`}
          style={{left:`calc(${n.x}% - ${n.hub?7:5}px)`,top:`calc(${n.y}% - ${n.hub?7:5}px)`}}
          onMouseEnter={()=>setHover(n)} onMouseLeave={()=>setHover(null)}/>
      ))}
      {hover&&(
        <div className="graph-tooltip" style={{
          left:hover.x<60?`calc(${hover.x}% + 14px)`:"auto",
          right:hover.x>=60?`calc(${100-hover.x}% + 14px)`:"auto",
          top:`calc(${hover.y}% - 30px)`,
        }}>
          <b style={{color:"var(--accent2)"}}>{hover.label}</b><br/>
          Items: {hover.chunks}<br/>
          Rel: {hover.edges}
        </div>
      )}
    </div>
  );
}

function DataAnalysis({ analysis, previews, onConfirm }) {
  const [selectedStrategy, setSelectedStrategy] = useState(analysis?.recommendation?.strategy || "recursive");

  return (
    <div style={{marginTop:10,background:"var(--panel)",border:"1px solid var(--accent)",borderRadius:10,padding:12, animation: "fadeIn 0.3s"}}>
      <div className="sb-label">Data Processing Analysis</div>
      
      <div className="preview-box" style={{fontSize: 11}}>
        <div className="prow"><span>Detected Type</span><span className="pval">{analysis?.recommendation?.reason}</span></div>
        <div className="prow"><span>Est. Tokens</span><span className="pval">{analysis?.estimated_tokens}</span></div>
        <div className="prow"><span>Avg Para Size</span><span className="pval">{analysis?.avg_tokens_per_paragraph} t</span></div>
      </div>

      <div className="sb-label" style={{marginTop:10}}>Chunking Preview</div>
      <div className="pill-group" style={{marginBottom: 10}}>
        {CHUNKERS.map(c => (
          <button key={c.id} className={`pill-opt ${selectedStrategy === c.id ? 'active' : ''}`} onClick={() => setSelectedStrategy(c.id)}>
            {c.label}
          </button>
        ))}
      </div>

      <div style={{maxHeight: 200, overflowY: 'auto', marginBottom: 10}}>
        {previews && previews[selectedStrategy] ? (
          previews[selectedStrategy].map((p, i) => (
            <div key={i} className="chunk-item" style={{fontSize: 10, opacity: 0.8}}>
              <div className="chunk-src">Preview Chunk {i+1} ({p.token_count} tokens)</div>
              <div className="chunk-text">{p.content}</div>
            </div>
          ))
        ) : (
          <div style={{padding: 20, textAlign: 'center', color: 'var(--muted)', fontSize: 11}}>No previews available for this strategy.</div>
        )}
      </div>

      <button className="embed-btn" onClick={() => onConfirm(selectedStrategy)}>
        ⚡ Ingest with {selectedStrategy} strategy
      </button>
    </div>
  );
}

export default function App() {
  const [sbOpen, setSbOpen] = useState(true);
  const [rpOpen, setRpOpen] = useState(true);
  const [mode, setMode] = useState("chat");
  const [provider, setProvider] = useState("groq");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("llama-3.3-70b-versatile");
  const [sources, setSources] = useState([]);
  const [ingestTab, setIngestTab] = useState("pdf");
  const [urlInput, setUrlInput] = useState("");
  const [embModel, setEmbModel] = useState("all-MiniLM-L6-v2");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({});
  const [analysisData, setAnalysisData] = useState(null);
  const [toast, setToast] = useState(null);
  const bottomRef = useRef(null);

  const showToast = (msg) => { setToast(msg); setTimeout(() => setToast(null), 3000); };

  useEffect(() => {
    fetchStats();
    const savedKey = localStorage.getItem("groq_api_key");
    if (savedKey) setApiKey(savedKey);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const fetchStats = async () => {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      setStats(data);
    } catch (e) { console.error(e); }
  };

  const handleConfig = async () => {
    try {
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model, api_key: apiKey })
      });
      localStorage.setItem("groq_api_key", apiKey);
      showToast("LLM Configured Successfully");
    } catch (e) { showToast("Config failed"); }
  };

  const handleAnalyze = async (file) => {
    const formData = new FormData();
    if (file) formData.append("file", file);
    else formData.append("url", urlInput);
    
    formData.append("source_type", ingestTab === "yt" ? "youtube" : ingestTab === "web" ? "website" : "pdf");

    showToast("Analyzing content...");
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await res.json();
      setAnalysisData({ ...data, file, url: urlInput, type: formData.get("source_type") });
    } catch (e) { showToast("Analysis failed"); }
  };

  const handleIngest = async (strategy) => {
    const formData = new FormData();
    if (analysisData.file) formData.append("file", analysisData.file);
    else formData.append("url", analysisData.url);
    
    formData.append("source_type", analysisData.type);
    formData.append("strategy", strategy);
    formData.append("embedding_model", embModel);

    showToast("Ingesting source...");
    setAnalysisData(null);
    try {
      await fetch("/api/ingest", { method: "POST", body: formData });
      showToast("Ingestion Complete");
      fetchStats();
      setUrlInput("");
    } catch (e) { showToast("Ingestion failed"); }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { id: Date.now(), role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input, mode, stream: true })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = { id: Date.now() + 1, role: "assistant", content: "" };
      setMessages(prev => [...prev, assistantMsg]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const token = line.slice(6);
            if (token === "[DONE]") break;
            assistantMsg.content += token;
            setMessages(prev => prev.map(m => m.id === assistantMsg.id ? { ...assistantMsg } : m));
          }
        }
      }
    } catch (e) { showToast("Query failed"); }
    setLoading(false);
  };

  return (
    <div className="shell">
      {/* Sidebar */}
      <div className={`sidebar ${sbOpen ? "" : "collapsed"}`}>
        <div style={{padding:"14px 16px 10px",borderBottom:"1px solid var(--border)"}}>
          <div style={{fontWeight:800,fontSize:16}}><span style={{color:"var(--accent)"}}>Anti</span>RAG</div>
        </div>

        <div className="sb-section">
          <div className="sb-label">Chat Mode</div>
          <div className="pill-group">
            {MODES.map(m => (
              <button key={m.id} className={`pill-opt ${mode === m.id ? 'active ' + m.cls : ''}`} onClick={() => setMode(m.id)}>{m.label}</button>
            ))}
          </div>
        </div>

        <div className="sb-section">
          <div className="sb-label">LLM Configuration</div>
          <input className="sb-input" type="password" placeholder="Groq API Key" value={apiKey} onChange={e => setApiKey(e.target.value)} />
          <button className="action-btn" style={{marginTop: 5}} onClick={handleConfig}>Apply Config</button>
        </div>

        <div className="sb-section">
          <div className="sb-label">Ingest Source</div>
          <div className="tab-row">
            {['pdf', 'yt', 'web'].map(t => (
              <div key={t} className={`tab ${ingestTab === t ? 'active' : ''}`} onClick={() => setIngestTab(t)}>{t.toUpperCase()}</div>
            ))}
          </div>
          <div className="tab-body">
            {ingestTab === 'pdf' ? (
              <input type="file" onChange={e => handleAnalyze(e.target.files[0])} />
            ) : (
              <div style={{display: 'flex', gap: 5}}>
                <input className="sb-input" placeholder="URL..." value={urlInput} onChange={e => setUrlInput(e.target.value)} />
                <button className="send-btn" style={{width: 30, height: 30}} onClick={() => handleAnalyze()}>→</button>
              </div>
            )}
          </div>

          {analysisData && (
            <DataAnalysis 
              analysis={analysisData.analysis} 
              previews={analysisData.previews} 
              onConfirm={handleIngest} 
            />
          )}
        </div>

        <div className="sb-section">
          <div className="sb-label">Embedding Model</div>
          {EMB_MODELS.map(m => (
            <div key={m.id} className={`emb-opt ${embModel === m.id ? 'active' : ''}`} onClick={() => setEmbModel(m.id)}>
              <span>{m.name}</span>
              <span className="emb-dim">{m.dim}d</span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Chat */}
      <div className="main">
        <div className="chat-header">
          <div className="header-title"><span>Anti</span>RAG</div>
          <span className="mode-badge mb-chat">{mode.toUpperCase()}</span>
          <div style={{flex: 1}} />
          <div className="status-dot" />
          <span style={{fontSize: 10, color: "var(--muted)"}}>{stats?.total_chunks || 0} Chunks</span>
        </div>

        <div className="messages">
          {messages.map(m => (
            <div key={m.id} className={`msg ${m.role}`}>
              <div className={`avatar ${m.role === 'user' ? 'av-user' : 'av-bot'}`}>{m.role === 'user' ? 'U' : '✦'}</div>
              <div className={`bubble bubble-${m.role}`}>{m.content}</div>
            </div>
          ))}
          {loading && <div className="loading-dots"><span/><span/><span/></div>}
          <div ref={bottomRef} />
        </div>

        <div className="input-bar">
          <textarea className="chat-input" placeholder="Type a message..." value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())} />
          <button className="send-btn" onClick={sendMessage}>↑</button>
        </div>
      </div>

      {/* Right Panel */}
      <div className={`right-panel ${rpOpen ? "" : "collapsed"}`}>
        <div className="rp-section">
          <div className="rp-label">Knowledge Graph</div>
          <MiniGraph stats={stats} />
        </div>
        <div className="rp-section">
          <div className="rp-label">System Stats</div>
          <div className="db-row"><span>Total Chunks</span><span className="db-num">{stats?.total_chunks}</span></div>
          {stats?.chunks?.dimensions && Object.entries(stats.chunks.dimensions).map(([dim, count]) => (
            <div key={dim} className="dim-row">
              <span className="dim-label">{dim}d</span>
              <div className="dim-bar" style={{width: `${(count/stats.total_chunks)*100}%`}} />
              <span className="dim-count">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
