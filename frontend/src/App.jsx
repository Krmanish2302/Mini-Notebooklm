import { useState, useRef, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
//  STYLES
// ─────────────────────────────────────────────────────────────────────────────
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;500;600;700;800&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#0a0a0f; --surface:#111118; --panel:#16161f;
  --border:#252535; --border2:#2e2e45;
  --accent:#7c6af7; --accent2:#c084fc;
  --green:#34d399; --amber:#fbbf24; --red:#f87171;
  --text:#e2e2f0; --muted:#6b6b8a; --dim:#3a3a52;
  --font-ui:'Syne',sans-serif; --font-mono:'DM Mono',monospace;
  --sb-w:288px; --rp-w:252px;
}

html,body,#root{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-ui);overflow:hidden}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--dim);border-radius:4px}

.shell{display:flex;height:100vh;overflow:hidden;position:relative}

/* sidebars */
.sidebar,.right-panel{
  height:100vh;display:flex;flex-direction:column;
  overflow-y:auto;overflow-x:hidden;flex-shrink:0;
  transition:width .22s cubic-bezier(.4,0,.2,1),min-width .22s cubic-bezier(.4,0,.2,1),opacity .18s;
}
.sidebar{width:var(--sb-w);min-width:var(--sb-w);background:var(--surface);border-right:1px solid var(--border)}
.right-panel{width:var(--rp-w);min-width:var(--rp-w);background:var(--surface);border-left:1px solid var(--border)}
.sidebar.collapsed,.right-panel.collapsed{width:0!important;min-width:0!important;opacity:0;pointer-events:none}

.sb-toggle{
  position:absolute;top:50%;z-index:30;width:20px;height:48px;
  background:var(--surface);border:1px solid var(--border2);
  border-radius:0 8px 8px 0;display:flex;align-items:center;justify-content:center;
  cursor:pointer;color:var(--muted);font-size:10px;
  transition:left .22s cubic-bezier(.4,0,.2,1),right .22s cubic-bezier(.4,0,.2,1),color .12s;
  transform:translateY(-50%);user-select:none;
}
.sb-toggle:hover{color:var(--accent);border-color:var(--accent)}
.sb-toggle.right{border-radius:8px 0 0 8px}

/* section primitives */
.sb-section{padding:14px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.sb-label{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}

/* pill groups */
.pill-group{display:flex;gap:4px;background:var(--panel);border-radius:9px;padding:3px}
.pill-opt{
  flex:1;padding:6px 4px;border-radius:6px;border:none;
  background:transparent;color:var(--muted);font-family:var(--font-ui);
  font-size:11px;font-weight:600;cursor:pointer;transition:all .14s;text-align:center;
}
.pill-opt.active{background:var(--surface);color:var(--accent);box-shadow:0 1px 4px rgba(0,0,0,.4)}
.pill-opt.active.amber{color:var(--amber)}
.pill-opt.active.green{color:var(--green)}

/* inputs */
.sb-input{
  width:100%;background:var(--panel);border:1px solid var(--border);
  border-radius:7px;padding:8px 10px;color:var(--text);
  font-family:var(--font-mono);font-size:12px;outline:none;transition:border-color .15s;
}
.sb-input:focus{border-color:var(--accent)}
.sb-input::placeholder{color:var(--muted)}

/* search results */
.search-result{
  display:flex;align-items:flex-start;gap:8px;padding:7px 8px;border-radius:7px;
  cursor:pointer;font-size:11px;transition:background .1s;margin-bottom:4px;
  border:1px solid transparent;
}
.search-result:hover{background:var(--panel);border-color:var(--border)}
.search-result.sel{background:rgba(124,106,247,.07);border-color:rgba(124,106,247,.3)}
.sr-title{font-weight:600;color:var(--text);margin-bottom:2px;line-height:1.3}
.sr-snip{color:var(--muted);font-size:10px}

/* tabs */
.tab-row{display:flex;border-bottom:1px solid var(--border)}
.tab{padding:7px 11px;font-size:11px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;transition:all .14s;white-space:nowrap}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-body{padding:10px;border:1px solid var(--border);border-top:none;border-radius:0 0 8px 8px;background:var(--panel)}

/* chips */
.chip-row{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}
.chip{
  padding:4px 9px;border-radius:20px;border:1px solid var(--border2);
  background:transparent;color:var(--muted);font-family:var(--font-ui);
  font-size:10px;cursor:pointer;transition:all .14s;
}
.chip:hover,.chip.active{border-color:var(--accent2);color:var(--accent2);background:rgba(192,132,252,.08)}
.chip:disabled{opacity:.4;cursor:default}

/* preview box */
.preview-box{
  background:var(--bg);border:1px solid var(--border2);border-radius:8px;
  padding:10px;font-family:var(--font-mono);font-size:10px;color:var(--muted);
  line-height:1.7;margin-bottom:8px;
}
.prow{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid var(--border)}
.prow:last-child{border-bottom:none}
.pval{color:var(--text);font-weight:500}

/* token stats table */
.token-table{width:100%;border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:8px;font-family:var(--font-mono);font-size:10px}
.token-table-hdr{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;background:var(--panel);padding:5px 8px;border-bottom:1px solid var(--border);font-size:9px;font-weight:700;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
.token-table-row{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;padding:5px 8px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .12s}
.token-table-row:last-child{border-bottom:none}
.token-table-row:hover{background:rgba(124,106,247,.05)}
.token-table-row.active-row{background:rgba(124,106,247,.08);color:var(--accent)}

/* embedding model options */
.emb-opt{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 9px;border-radius:7px;border:1px solid var(--border);
  cursor:pointer;margin-bottom:5px;font-size:11px;transition:all .14s;
}
.emb-opt:hover{border-color:var(--border2)}
.emb-opt.active{border-color:var(--accent);background:rgba(124,106,247,.08)}
.emb-dim{font-family:var(--font-mono);font-size:10px;color:var(--muted)}
.emb-note{font-size:9px;color:var(--muted);font-family:var(--font-mono);margin-top:1px}

/* EMBED CTA button */
.embed-btn{
  width:100%;padding:10px;border-radius:9px;border:none;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;font-family:var(--font-ui);font-size:12px;font-weight:700;
  cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;
  letter-spacing:.04em;transition:opacity .15s;margin-top:8px;
}
.embed-btn:hover{opacity:.88}
.embed-btn:disabled{opacity:.4;cursor:default}

/* progress */
.prog-wrap{margin-top:8px;background:var(--bg);border:1px solid var(--border2);border-radius:8px;padding:10px}
.prog-lbl{font-size:10px;color:var(--muted);font-family:var(--font-mono);margin-bottom:6px}
.prog-bar-bg{height:4px;background:var(--border);border-radius:4px;overflow:hidden}
.prog-bar{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:4px;transition:width .35s ease}
.prog-step{font-size:9px;color:var(--accent);font-family:var(--font-mono);margin-top:5px}

/* source rows */
.source-row{
  display:flex;align-items:flex-start;gap:8px;
  padding:8px 9px;border-radius:8px;border:1px solid var(--border);
  margin-bottom:6px;font-size:11px;background:var(--panel);animation:fadeIn .2s;
}
.src-icon{font-size:14px;flex-shrink:0;margin-top:1px}
.src-meta{flex:1;min-width:0}
.src-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}
.src-badge{display:inline-block;padding:1px 5px;border-radius:4px;font-size:9px;font-family:var(--font-mono);font-weight:500;margin-right:3px}
.badge-ready{background:rgba(52,211,153,.12);color:var(--green)}
.badge-pending{background:rgba(251,191,36,.1);color:var(--amber)}
.badge-processing{background:rgba(124,106,247,.12);color:var(--accent)}
.src-stats{color:var(--muted);font-family:var(--font-mono);font-size:10px;margin-top:2px}
.src-del{background:none;border:none;color:var(--dim);cursor:pointer;padding:2px;font-size:13px;border-radius:4px;transition:color .1s;flex-shrink:0}
.src-del:hover{color:var(--red)}

/* totals strip */
.totals-strip{display:flex;border:1px solid var(--border2);border-radius:8px;overflow:hidden;font-family:var(--font-mono);font-size:10px;margin-top:8px}
.total-cell{flex:1;padding:7px 8px;text-align:center;border-right:1px solid var(--border)}
.total-cell:last-child{border-right:none}
.total-num{color:var(--green);font-size:14px;font-weight:500;display:block;line-height:1.2}
.total-lbl{color:var(--muted);font-size:9px}

/* new chat */
.new-chat-btn{
  width:100%;padding:9px;border-radius:8px;
  background:rgba(124,106,247,.15);border:1px solid rgba(124,106,247,.3);
  color:var(--accent);font-family:var(--font-ui);font-size:12px;font-weight:700;
  cursor:pointer;transition:all .14s;display:flex;align-items:center;justify-content:center;gap:8px;
}
.new-chat-btn:hover{background:rgba(124,106,247,.25)}

/* history */
.hist-item{padding:7px 10px;border-radius:7px;cursor:pointer;font-size:12px;margin-bottom:3px;transition:background .1s;display:flex;align-items:center;gap:7px}
.hist-item:hover{background:var(--panel)}
.hist-item.active{background:rgba(124,106,247,.12);color:var(--accent)}
.hist-dot{width:5px;height:5px;border-radius:50%;background:var(--green);flex-shrink:0}
.hist-dot.old{background:var(--dim)}

/* main */
.main{flex:1;display:flex;flex-direction:column;height:100vh;overflow:hidden;min-width:0}
.chat-header{padding:12px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;background:var(--surface);flex-shrink:0}
.header-title{font-weight:800;font-size:15px;letter-spacing:-.02em}
.header-title span{color:var(--accent)}
.mode-badge{padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.06em}
.mb-chat{background:rgba(124,106,247,.15);color:var(--accent)}
.mb-deep{background:rgba(251,191,36,.12);color:var(--amber)}
.mb-study{background:rgba(52,211,153,.12);color:var(--green)}

/* messages */
.messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;animation:fadeIn .2s}
.msg.user{flex-direction:row-reverse}
.avatar{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
.av-user{background:rgba(124,106,247,.2)}
.av-bot{background:rgba(52,211,153,.15)}
.bubble{max-width:70%;padding:12px 15px;border-radius:14px;font-size:13.5px;line-height:1.65}
.bubble-user{background:rgba(124,106,247,.11);border:1px solid rgba(124,106,247,.2)}
.bubble-bot{background:var(--panel);border:1px solid var(--border)}

.cite-tag{
  display:inline-flex;align-items:center;
  background:rgba(124,106,247,.15);color:var(--accent);
  padding:1px 6px;border-radius:4px;font-size:10px;font-family:var(--font-mono);
  cursor:pointer;margin:0 2px;transition:background .1s;border:1px solid rgba(124,106,247,.3);
}
.cite-tag:hover{background:rgba(124,106,247,.3)}

.chunks-section{margin-top:10px;border-top:1px solid var(--border);padding-top:10px}
.chunks-hdr{font-size:9px;color:var(--muted);font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.chunk-item{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px 11px;margin-bottom:6px;font-family:var(--font-mono);font-size:11px;color:var(--muted);line-height:1.6}
.chunk-src{color:var(--accent2);font-weight:500;margin-bottom:4px;font-size:10px}
.chunk-text{color:var(--text)}

/* loading */
.loading-dots span{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--muted);margin:0 2px;animation:bounce .9s infinite}
.loading-dots span:nth-child(2){animation-delay:.15s}
.loading-dots span:nth-child(3){animation-delay:.3s}

/* input */
.input-bar{padding:12px 18px;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:9px;align-items:flex-end;flex-shrink:0}
.chat-input{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:11px 15px;color:var(--text);font-family:var(--font-ui);font-size:14px;outline:none;resize:none;min-height:46px;max-height:130px;line-height:1.5;transition:border-color .14s}
.chat-input:focus{border-color:var(--accent)}
.chat-input::placeholder{color:var(--muted)}
.send-btn{width:42px;height:42px;border-radius:11px;border:none;background:var(--accent);color:#fff;cursor:pointer;font-size:17px;display:flex;align-items:center;justify-content:center;transition:opacity .14s;flex-shrink:0}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.38;cursor:default}

.statusbar{padding:3px 16px;background:var(--bg);border-top:1px solid var(--border);display:flex;gap:10px;align-items:center;flex-shrink:0;font-family:var(--font-mono);font-size:9px;color:var(--muted)}
.status-dot{width:5px;height:5px;border-radius:50%;background:var(--green);display:inline-block}

/* right panel */
.rp-section{padding:14px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.rp-label{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
.graph-canvas{height:190px;background:var(--bg);border:1px solid var(--border);border-radius:8px;position:relative;overflow:hidden}
.graph-node{position:absolute;width:10px;height:10px;border-radius:50%;background:var(--accent);cursor:pointer;transition:transform .2s;border:2px solid rgba(124,106,247,.35)}
.graph-node:hover{transform:scale(1.7);z-index:10}
.graph-node.hub{width:14px;height:14px;background:var(--accent2)}
.graph-tooltip{position:absolute;background:var(--panel);border:1px solid var(--border2);border-radius:7px;padding:7px 9px;font-size:10px;font-family:var(--font-mono);pointer-events:none;z-index:20;max-width:140px;color:var(--text);line-height:1.5;box-shadow:0 4px 12px rgba(0,0,0,.5)}
.db-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);font-size:11px}
.db-row:last-child{border-bottom:none}
.db-num{font-family:var(--font-mono);color:var(--green);font-size:13px;font-weight:500}
.dim-row{display:flex;align-items:center;gap:6px;padding:5px 7px;border-radius:6px;border:1px solid var(--border);margin-bottom:4px;font-size:10px;font-family:var(--font-mono)}
.dim-bar{height:3px;border-radius:3px;background:var(--accent);flex:1;opacity:.7}
.dim-label{color:var(--muted);width:36px;flex-shrink:0}
.dim-count{color:var(--text);width:28px;text-align:right;flex-shrink:0}
.param-row{margin-bottom:11px}
.param-label{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:5px}
.param-label span{color:var(--text);font-family:var(--font-mono);font-weight:500}
input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer;height:3px;border-radius:3px}
.action-btn{width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--border2);background:transparent;color:var(--muted);font-family:var(--font-ui);font-size:11px;font-weight:600;cursor:pointer;transition:all .14s;margin-bottom:5px;text-align:left;display:flex;align-items:center;gap:7px}
.action-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(124,106,247,.06)}
.action-btn.danger:hover{border-color:var(--red);color:var(--red);background:rgba(248,113,113,.06)}
.hdr-btn{padding:4px 10px;border-radius:7px;border:1px solid var(--border2);background:transparent;color:var(--muted);font-family:var(--font-ui);font-size:10px;font-weight:600;cursor:pointer;transition:all .14s}
.hdr-btn:hover{border-color:var(--accent);color:var(--accent)}

.toast{position:fixed;bottom:18px;right:18px;background:var(--panel);border:1px solid var(--green);border-radius:10px;padding:9px 15px;font-size:12px;color:var(--green);z-index:999;animation:slideUp .22s;box-shadow:0 4px 16px rgba(0,0,0,.5)}
.toast.err{border-color:var(--red);color:var(--red)}

@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes bounce{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}
`;

// ─────────────────────────────────────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const MODES = [
  {id:"chat",  label:"Chat",     cls:""},
  {id:"deep",  label:"Research", cls:"amber"},
  {id:"study", label:"Study",    cls:"green"},
];

// Default embedding models — overwritten by GET /api/embedding-models on mount
const DEFAULT_EMB_MODELS = [
  {id:"minilm",  name:"all-MiniLM-L6-v2",      dim:384,  tokens:256,  note:"Local · fastest"},
  {id:"mpnet",   name:"all-mpnet-base-v2",      dim:768,  tokens:384,  note:"Local · balanced"},
  {id:"e5large", name:"e5-large-v2",            dim:1024, tokens:512,  note:"Local · most accurate"},
  {id:"oai-sm",  name:"text-embedding-3-small", dim:1536, tokens:8191, note:"OpenAI API key required"},
  {id:"oai-lg",  name:"text-embedding-3-large", dim:3072, tokens:8191, note:"OpenAI API key required"},
];

const CHUNKERS = [
  {id:"paragraph",    label:"Paragraph"},
  {id:"page",         label:"Page"},
  {id:"recursive",    label:"Recursive"},
  {id:"semantic",     label:"Semantic"},
  {id:"hierarchical", label:"Hierarchical"},
];

// Fallback static estimates used only when /api/analyze is unavailable
const CHUNK_EST = {paragraph:47,page:12,recursive:39,hierarchical:28,semantic:31};
const SRC_ICONS  = {pdf:"📄",youtube:"▶️",website:"🌐",text:"📝",image:"🖼️",csv:"📊",video:"🎬"};

// ─────────────────────────────────────────────────────────────────────────────
//  MINI GRAPH  — fetches real KG from GET /api/graph, falls back to demo data
// ─────────────────────────────────────────────────────────────────────────────
function MiniGraph() {
  const [hover, setHover] = useState(null);
  const [graphData, setGraphData] = useState(null);

  useEffect(() => {
    fetch("/api/graph")
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.nodes && data.nodes.length > 0) {
          setGraphData(data);
        }
      })
      .catch(() => {});
  }, []);

  // Layout: spread nodes evenly in the canvas using simple circular placement
  const buildLayout = (nodes) => {
    const cx = 50, cy = 50, r = 35;
    return nodes.map((n, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
      return {
        ...n,
        x: Math.round(cx + r * Math.cos(angle)),
        y: Math.round(cy + r * Math.sin(angle)),
        hub: (n.type === "source" || (n.edges_count || 0) >= 3),
        label: n.id?.slice(0, 18) || `node-${i}`,
      };
    });
  };

  // Demo fallback nodes + edges
  const demoNodes = [
    {x:50,y:38,hub:true, label:"Transformers",id:"Transformers"},
    {x:24,y:68,hub:false,label:"Attention",   id:"Attention"},
    {x:76,y:63,hub:false,label:"FAISS",       id:"FAISS"},
    {x:50,y:84,hub:false,label:"Chunking",    id:"Chunking"},
    {x:14,y:42,hub:false,label:"Embeddings",  id:"Embeddings"},
    {x:86,y:33,hub:false,label:"BM25",        id:"BM25"},
  ];
  const demoEdges = [[0,1],[0,2],[0,3],[0,4],[1,4],[2,5],[1,3]];

  let nodes, edgePairs;
  if (graphData) {
    const laid = buildLayout(graphData.nodes.slice(0, 20));
    const idxById = Object.fromEntries(laid.map((n, i) => [n.id, i]));
    nodes = laid;
    edgePairs = graphData.edges
      .slice(0, 40)
      .map(e => [idxById[e.from], idxById[e.to]])
      .filter(([a, b]) => a !== undefined && b !== undefined);
  } else {
    nodes = demoNodes;
    edgePairs = demoEdges;
  }

  return (
    <div className="graph-canvas">
      <svg style={{position:"absolute",top:0,left:0,width:"100%",height:"100%"}}>
        {edgePairs.map(([a,b],i) => (
          <line key={i}
            x1={`${nodes[a].x}%`} y1={`${nodes[a].y}%`}
            x2={`${nodes[b].x}%`} y2={`${nodes[b].y}%`}
            stroke="var(--dim)" strokeWidth="1"/>
        ))}
      </svg>
      {nodes.map((n,i) => (
        <div key={i}
          className={`graph-node${n.hub?" hub":""}`}
          style={{
            left:`calc(${n.x}% - ${n.hub?7:5}px)`,
            top:`calc(${n.y}% - ${n.hub?7:5}px)`,
          }}
          onMouseEnter={() => setHover(n)}
          onMouseLeave={() => setHover(null)}
        />
      ))}
      {hover && (
        <div className="graph-tooltip" style={{
          left:  hover.x < 60 ? `calc(${hover.x}% + 14px)` : "auto",
          right: hover.x >= 60 ? `calc(${100-hover.x}% + 14px)` : "auto",
          top:   `calc(${hover.y}% - 30px)`,
        }}>
          <b style={{color:"var(--accent2)"}}>{hover.label}</b><br/>
          {hover.type && <>Type: {hover.type}<br/></>}
          {hover.chunks && <>Chunks: {hover.chunks}<br/></>}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  EMBED FLOW  — wired to real /api/analyze + /api/ingest
// ─────────────────────────────────────────────────────────────────────────────
function EmbedFlow({pendingSources, chunker, setChunker, embModel, setEmbModel,
                    embModels, onDone, analysisData, analyzing}) {
  const [phase,    setPhase]    = useState("idle");
  const [progress, setProgress] = useState(0);
  const [stepLabel,setStepLabel]= useState("");

  const emb = embModels.find(e => e.id === embModel) || embModels[0];

  // Real per-strategy stats if available, else static estimate
  const stratData  = analysisData?.[chunker];
  const totalChunks = stratData
    ? stratData.total_chunks
    : CHUNK_EST[chunker] * Math.max(pendingSources.length, 1);
  const totalTokens = totalChunks * (emb?.tokens || 256);

  const run = async () => {
    if (!pendingSources.length) return;
    setPhase("analyzing"); setProgress(10); setStepLabel("Sending to ingest pipeline…");
    try {
      // Ingest each pending source sequentially (each may be a file or URL)
      let lastResult = null;
      for (const src of pendingSources) {
        const form = new FormData();
        form.append("source_type",       src.type);
        form.append("chunking_strategy", chunker);
        // Map UI model id → actual model name expected by pipeline
        form.append("embedding_model",   emb?.name || emb?.id || embModel);

        if (src._file) {
          // PDF / file upload: send the actual File blob
          form.append("file", src._file, src._file.name);
        } else if (src._url || src.type === "website" || src.type === "youtube") {
          form.append("url", src._url || src.name);
        } else if (src.type === "text" && src._text) {
          // Plain text: wrap in a Blob and send as a .txt file
          form.append("file", new Blob([src._text], {type:"text/plain"}), "paste.txt");
        }

        setProgress(30); setStepLabel(`Chunking · ${chunker} strategy…`);
        const res = await fetch("/api/ingest", {method:"POST", body:form});
        setProgress(65); setStepLabel(`Embedding with ${emb?.name}…`);
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`Ingest failed (${res.status}): ${errText}`);
        }
        lastResult = await res.json();
        setProgress(90); setStepLabel("Writing to FAISS vector store…");
      }
      await new Promise(r => setTimeout(r, 300));
      setProgress(100); setPhase("done"); setStepLabel("✓ All sources embedded");
      onDone(embModel, chunker, lastResult?.result?.chunk_count ?? totalChunks);
    } catch (err) {
      setPhase("idle"); setProgress(0); setStepLabel("");
      onDone(null, null, 0, err.message);
    }
  };

  if (pendingSources.length === 0) return null;

  return (
    <div style={{marginTop:10,background:"var(--bg)",border:"1px solid var(--border2)",borderRadius:10,padding:12}}>
      <div className="sb-label">Queued Sources</div>
      {pendingSources.map(s => (
        <div key={s.id} style={{marginBottom:7,padding:"8px 10px",background:"var(--panel)",borderRadius:8,border:"1px solid var(--border)"}}>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:5}}>
            <span style={{fontSize:13}}>{SRC_ICONS[s.type]||"📁"}</span>
            <span style={{fontSize:11,fontWeight:600,flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.name}</span>
            <span className="src-badge badge-pending">pending</span>
          </div>
          <div style={{fontFamily:"var(--font-mono)",fontSize:10,color:"var(--muted)"}}>
            type: <span style={{color:"var(--text)"}}>{s.type}</span>
            {" · "}dim: <span style={{color:"var(--text)"}}>{emb?.dim}</span>
          </div>
        </div>
      ))}

      <div className="sb-label" style={{marginTop:10}}>Chunking Strategy</div>
      <div className="chip-row">
        {CHUNKERS.map(c => (
          <button key={c.id} className={`chip${chunker===c.id?" active":""}`}
            disabled={phase!=="idle"} onClick={() => setChunker(c.id)}>
            {c.label}
          </button>
        ))}
      </div>

      {/* Spinner while /api/analyze is running */}
      {analyzing && (
        <div style={{fontFamily:"var(--font-mono)",fontSize:10,color:"var(--muted)",
                     padding:"6px 0",display:"flex",alignItems:"center",gap:6}}>
          <span style={{display:"inline-block",animation:"spin 1s linear infinite"}}>⟳</span>
          Analysing document structure…
        </div>
      )}

      {/* Real per-strategy token table from /api/analyze → token_stats */}
      {analysisData && !analyzing && (
        <div className="token-table">
          <div className="token-table-hdr">
            <span>Strategy</span>
            <span>Avg tok</span>
            <span style={{color:"var(--green)"}}>Min</span>
            <span style={{color:"var(--amber)"}}>Max</span>
          </div>
          {Object.entries(analysisData).map(([key, val]) => (
            <div key={key}
              className={`token-table-row${chunker===key?" active-row":""}`}
              onClick={() => phase==="idle" && setChunker(key)}>
              <span style={{textTransform:"capitalize",fontWeight:chunker===key?700:400}}>{key}</span>
              <span>{val.avg ?? val.avg_tokens ?? "—"}</span>
              <span style={{color:"var(--green)"}}>{val.min ?? val.min_tokens ?? "—"}</span>
              <span style={{color:"var(--amber)"}}>{val.max ?? val.max_tokens ?? "—"}</span>
            </div>
          ))}
        </div>
      )}

      {/* Preview summary box */}
      <div className="preview-box">
        <div className="prow"><span>Est. chunks</span>  <span className="pval">{totalChunks}</span></div>
        <div className="prow"><span>Tokens/chunk</span> <span className="pval">~{emb?.tokens}</span></div>
        <div className="prow"><span>Total tokens</span> <span className="pval">{totalTokens.toLocaleString()}</span></div>
        <div className="prow"><span>Vector dim</span>   <span className="pval">{emb?.dim}</span></div>
      </div>

      <div className="sb-label">Embedding Model</div>
      {embModels.map(e => (
        <div key={e.id}
          className={`emb-opt${embModel===e.id?" active":""}`}
          onClick={() => phase==="idle" && setEmbModel(e.id)}
          style={{cursor:phase!=="idle"?"default":"pointer"}}>
          <div>
            <div style={{fontWeight:600,fontSize:10}}>{e.name}</div>
            <div className="emb-dim">{e.dim} dims · max {e.tokens} tok</div>
            <div className="emb-note">{e.note}</div>
          </div>
          {embModel===e.id && <span style={{color:"var(--accent)",fontSize:12}}>✓</span>}
        </div>
      ))}

      {phase!=="idle" && (
        <div className="prog-wrap">
          <div className="prog-lbl">{phase.charAt(0).toUpperCase()+phase.slice(1)}…</div>
          <div className="prog-bar-bg"><div className="prog-bar" style={{width:`${progress}%`}}/></div>
          <div className="prog-step">{stepLabel}</div>
        </div>
      )}

      <button className="embed-btn"
        disabled={phase!=="idle" && phase!=="done"}
        onClick={run}>
        {phase==="idle"   && <><span>⚡</span>Embed &amp; Ingest {pendingSources.length} source{pendingSources.length>1?"s":""}</>}
        {phase!=="idle" && phase!=="done" && <><span style={{display:"inline-block",animation:"spin 1s linear infinite"}}>⟳</span>Processing…</>}
        {phase==="done"   && <><span>✓</span>Done — sources ready in store</>}
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  APP
// ─────────────────────────────────────────────────────────────────────────────
function useToast() {
  const [t,   setT]   = useState(null);
  const [err, setErr] = useState(false);
  const show = (msg, isErr=false) => {
    setT(msg); setErr(isErr);
    setTimeout(() => { setT(null); setErr(false); }, 2800);
  };
  return [t, err, show];
}

export default function App() {
  const [sbOpen,   setSbOpen]   = useState(true);
  const [rpOpen,   setRpOpen]   = useState(true);
  const [mode,     setMode]     = useState("chat");
  const [provider, setProvider] = useState("groq");
  const [apiKey,   setApiKey]   = useState("");
  const [model,    setModel]    = useState("llama-3.3-70b-versatile");
  const [ollamaUrl,setOllamaUrl]= useState("http://localhost:11434");
  const [searchQ,  setSearchQ]  = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedSR,    setSelectedSR]    = useState({});

  // Sources: starts empty, populated from GET /api/sources on mount
  const [sources,        setSources]        = useState([]);
  const [pendingSources, setPendingSources] = useState([]);
  const [ingestTab,      setIngestTab]      = useState("pdf");
  const [ytUrl,          setYtUrl]          = useState("");
  const [webUrl,         setWebUrl]         = useState("");
  const [pasteText,      setPasteText]      = useState("");
  const fileRef = useRef(null);

  const [chunker,  setChunker]  = useState("hierarchical");
  const [embModel, setEmbModel] = useState("minilm");
  // Embedding model catalogue — overwritten by GET /api/embedding-models
  const [embModels, setEmbModels] = useState(DEFAULT_EMB_MODELS);

  // Real per-strategy token analysis data from /api/analyze → token_stats key
  const [analysisData, setAnalysisData] = useState(null);
  const [analyzing,    setAnalyzing]    = useState(false);

  // Stats from GET /api/stats (refreshed after ingest/delete)
  const [stats, setStats] = useState({total_chunks:0, total_sources:0, graph:{nodes:0,edges:0}, chunks:{dimensions:{}}});

  const [history,       setHistory]       = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages,      setMessages]      = useState([]);
  const [input,         setInput]         = useState("");
  const [loading,       setLoading]       = useState(false);
  const [temp,          setTemp]          = useState(0.7);
  const [topP,          setTopP]          = useState(0.9);
  const [topK,          setTopK]          = useState(40);
  const [maxTokens,     setMaxTokens]     = useState(1024);
  const [toast, toastErr, showToast]      = useToast();
  const bottomRef = useRef(null);

  // ── on mount: load sources + embedding models ──────────────────────────────
  useEffect(() => {
    // Fetch persisted sources
    fetch("/api/sources")
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.sources?.length) {
          setSources(data.sources.map(s => ({
            id:       s.source_id || s.id,
            type:     s.source_type || s.type || "text",
            name:     s.name || s.source_name || s.id,
            chunks:   s.chunk_count ?? s.chunks ?? 0,
            vectors:  s.chunk_count ?? s.vectors ?? 0,
            embModel: s.embedding_model || "minilm",
            status:   "ready",
          })));
        }
      })
      .catch(() => {});

    // Fetch live embedding model catalogue
    fetch("/api/embedding-models")
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.models?.length) {
          // Map api model objects → UI shape (add a stable id)
          setEmbModels(data.models.map(m => ({
            id:     m.name.replace(/[^a-z0-9]/gi, "-").toLowerCase(),
            name:   m.name,
            dim:    m.dim,
            tokens: m.max_tokens,
            label:  m.label,
            note:   m.note,
          })));
        }
      })
      .catch(() => {});

    // Fetch initial stats
    refreshStats();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({behavior:"smooth"});
  }, [messages, loading]);

  // ── refresh stats from GET /api/stats ─────────────────────────────────────
  const refreshStats = useCallback(() => {
    fetch("/api/stats")
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setStats(data); })
      .catch(() => {});
  }, []);

  // Derived display values (fall back to React state when stats not yet loaded)
  const totalChunks  = stats.total_chunks  || sources.reduce((a,s) => a+s.chunks,  0);
  const totalVectors = stats.total_chunks  || sources.reduce((a,s) => a+s.vectors, 0);
  const totalSources = stats.total_sources || sources.length;

  // Dimension breakdown from live stats
  const dimBreakdown = Object.entries(stats.chunks?.dimensions || {}).map(([dim, count]) => ({
    dim: Number(dim), count,
  }));

  const selEmb      = embModels.find(e => e.id === embModel) || embModels[0];
  const modeBadge   = {chat:"mb-chat", deep:"mb-deep", study:"mb-study"}[mode];
  const modeLabel   = {chat:"Chat",    deep:"Deep Research", study:"Study"}[mode];

  // ── web search (mock) ──────────────────────────────────────────────────────
  const doSearch = () => {
    if (!searchQ.trim()) return;
    setSearchResults([
      {id:"r1",title:"Retrieval Augmented Generation – DuckDuckGo",    snippet:"RAG combines retrieval with language model generation for grounded answers…"},
      {id:"r2",title:"Hugging Face – FAISS documentation",             snippet:"FAISS is a library for efficient similarity search and clustering of dense vectors…"},
      {id:"r3",title:"LangChain – RAG conceptual guide",               snippet:"Learn to build production RAG pipelines with hybrid retrieval and reranking…"},
    ]);
  };

  const ingestSelected = () => {
    const sel = Object.entries(selectedSR).filter(([,v]) => v).map(([k]) => k);
    if (!sel.length) return showToast("Select at least one result first", true);
    const news = sel.map(id => {
      const r = searchResults.find(x => x.id === id);
      return {id:`src_${id}`, type:"website", name:r.title.slice(0,42),
              chunks:0, vectors:0, embModel, status:"pending", _url:r.url||r.title};
    });
    setPendingSources(p => [...p, ...news]);
    setSelectedSR({}); setSearchResults([]); setSearchQ("");
    showToast(`${news.length} source(s) queued`);
  };

  // ── /api/analyze — called after file is queued ─────────────────────────────
  // FIX #1: read data.token_stats (not data.strategies)
  const analyzeFile = async (file) => {
    setAnalyzing(true);
    setAnalysisData(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res  = await fetch("/api/analyze", {method:"POST", body:form});
      if (!res.ok) throw new Error(`analyze ${res.status}`);
      const data = await res.json();
      // api.py returns: { token_stats: { paragraph:{avg,min,max,total_chunks}, ... } }
      setAnalysisData(data.token_stats ?? null);
    } catch (err) {
      console.warn("analyze fallback:", err.message);
      setAnalysisData(null);
    } finally {
      setAnalyzing(false);
    }
  };

  const onFile = (e) => {
    const f = e.target.files?.[0]; if (!f) return;
    const ext  = f.name.split(".").pop().toLowerCase();
    const type = {pdf:"pdf",csv:"csv",png:"image",jpg:"image",mp4:"video"}[ext] || "text";
    setPendingSources(p => [...p, {
      id:`src_${Date.now()}`, type, name:f.name,
      chunks:0, vectors:0, embModel, status:"pending", _file:f,
    }]);
    e.target.value = "";
    showToast(`${f.name} queued — analysing…`);
    analyzeFile(f);
  };

  const addUrl = () => {
    const url = ingestTab==="yt" ? ytUrl : webUrl;
    if (!url.trim()) return;
    const type = ingestTab==="yt" ? "youtube" : "website";
    setPendingSources(p => [...p, {
      id:`src_${Date.now()}`, type, name:url.slice(0,44),
      chunks:0, vectors:0, embModel, status:"pending", _url:url,
    }]);
    ingestTab==="yt" ? setYtUrl("") : setWebUrl("");
    showToast("Source queued");
  };

  const addPaste = () => {
    if (!pasteText.trim()) return;
    setPendingSources(p => [...p, {
      id:`src_${Date.now()}`, type:"text", name:"Pasted text",
      chunks:0, vectors:0, embModel, status:"pending", _text:pasteText,
    }]);
    setPasteText(""); showToast("Text queued");
  };

  // ── EmbedFlow callback ─────────────────────────────────────────────────────
  const handleEmbedDone = (emb, chk, count, errMsg) => {
    if (errMsg) { showToast(`⚠ Ingest failed: ${errMsg}`, true); return; }
    const chunkCount = count || CHUNK_EST[chk] || 0;
    const done = pendingSources.map(s => ({
      ...s, status:"ready", embModel:emb, chunks:chunkCount, vectors:chunkCount,
    }));
    setSources(s => [...s, ...done]);
    setPendingSources([]);
    setAnalysisData(null);
    refreshStats();           // ← pull real numbers from backend
    showToast(`✓ ${done.length} source(s) embedded`);
  };

  // ── delete source: DELETE /api/sources/:id → FAISS + SQLite + state ────────
  const deleteSource = async (id) => {
    try {
      const res = await fetch(`/api/sources/${id}`, {method:"DELETE"});
      if (!res.ok) throw new Error(`${res.status}`);
      setSources(s => s.filter(x => x.id !== id));
      refreshStats();
      showToast("Removed from DB + vector store");
    } catch (err) {
      showToast(`⚠ Delete failed (${err.message})`, true);
    }
  };

  // ── sendMessage: real SSE streaming from POST /api/query/stream ────────────
  // FIX #3: replaces the old mock setTimeout with actual SSE
  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userText = input.trim();
    const userMsg  = {id:`m${Date.now()}`, role:"user", content:userText};
    setMessages(m => [...m, userMsg]);
    setInput("");
    setLoading(true);

    const botId = `m${Date.now()+1}`;
    // Insert an empty bot message immediately so the bubble appears
    setMessages(m => [...m, {id:botId, role:"assistant", content:"", chunks:[], citations:[]}]);

    try {
      const res = await fetch("/api/query/stream", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({query:userText, mode, stream:true}),
      });
      if (!res.ok) throw new Error(`Query failed: ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = "";

      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream:true});
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === "token") {
              setMessages(m => m.map(msg =>
                msg.id === botId
                  ? {...msg, content: msg.content + evt.content}
                  : msg
              ));
            } else if (evt.type === "metadata") {
              const chunks   = evt.chunks || evt.retrieved_chunks || [];
              const citations= evt.citations || evt.sources_used || [];
              setMessages(m => m.map(msg =>
                msg.id === botId ? {...msg, chunks, citations} : msg
              ));
            } else if (evt.type === "error") {
              throw new Error(evt.detail);
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      setMessages(m => m.map(msg =>
        msg.id === botId
          ? {...msg, content:`⚠ Error: ${err.message}`}
          : msg
      ));
    } finally {
      setLoading(false);
    }
  };

  const newChat = () => {
    fetch("/api/new-chat", {method:"POST"}).catch(() => {});
    const id = `h${Date.now()}`;
    setHistory(h => [{id, title:"New conversation"}, ...h]);
    setActiveSession(id);
    setMessages([]);
    showToast("New chat started");
  };

  const loadSession = (id) => {
    setActiveSession(id);
    setMessages([]);
  };

  // ── mode switch: POST /api/mode ────────────────────────────────────────────
  const handleModeChange = (m) => {
    setMode(m);
    fetch("/api/mode", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mode:m}),
    }).catch(() => {});
  };

  return (
    <>
      <style>{STYLE}</style>
      <div className="shell">

        {/* ── LEFT SIDEBAR ── */}
        <div className={`sidebar${sbOpen?"":" collapsed"}`}>
          <div style={{padding:"14px 16px 10px",borderBottom:"1px solid var(--border)",flexShrink:0}}>
            <div style={{fontWeight:800,fontSize:16,letterSpacing:"-.02em"}}>
              <span style={{color:"var(--accent)"}}>Mini</span>NotebookLM
            </div>
            <div style={{fontSize:10,color:"var(--muted)",fontFamily:"var(--font-mono)"}}>local · private · fast</div>
          </div>

          {/* Chat mode */}
          <div className="sb-section">
            <div className="sb-label">Chat Mode</div>
            <div className="pill-group">
              {MODES.map(m => (
                <button key={m.id}
                  className={`pill-opt${mode===m.id?` active${m.cls?" "+m.cls:""}`:""}`}
                  onClick={() => handleModeChange(m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* LLM provider */}
          <div className="sb-section">
            <div className="sb-label">LLM Provider</div>
            <div className="pill-group" style={{marginBottom:10}}>
              <button className={`pill-opt${provider==="groq"?" active":""}`}   onClick={() => setProvider("groq")}>Groq</button>
              <button className={`pill-opt${provider==="ollama"?" active":""}`} onClick={() => setProvider("ollama")}>Ollama</button>
            </div>
            {provider==="groq" && (
              <>
                <input className="sb-input" type="password" placeholder="API Key…"    value={apiKey}   onChange={e => setApiKey(e.target.value)}/>
                <input className="sb-input" placeholder="Model name"                  value={model}    onChange={e => setModel(e.target.value)} style={{marginTop:6}}/>
              </>
            )}
            {provider==="ollama" && (
              <input className="sb-input" placeholder="http://localhost:11434" value={ollamaUrl} onChange={e => setOllamaUrl(e.target.value)}/>
            )}
          </div>

          {/* Web search */}
          <div className="sb-section">
            <div className="sb-label">Web Search</div>
            <div style={{display:"flex",gap:6,marginBottom:8}}>
              <input className="sb-input" placeholder="Search query…" value={searchQ}
                onChange={e => setSearchQ(e.target.value)}
                onKeyDown={e => e.key==="Enter" && doSearch()} style={{flex:1}}/>
              <button className="send-btn" style={{width:32,height:32,borderRadius:8,fontSize:12}} onClick={doSearch}>↵</button>
            </div>
            {searchResults.map(r => (
              <div key={r.id} className={`search-result${selectedSR[r.id]?" sel":""}`}
                onClick={() => setSelectedSR(s => ({...s,[r.id]:!s[r.id]}))}>
                <input type="checkbox" readOnly checked={!!selectedSR[r.id]}
                  style={{marginTop:2,accentColor:"var(--accent)",flexShrink:0}}/>
                <div>
                  <div className="sr-title">{r.title}</div>
                  <div className="sr-snip">{r.snippet}</div>
                </div>
              </div>
            ))}
            {searchResults.length>0 && (
              <button className="action-btn" style={{marginTop:4}} onClick={ingestSelected}>
                ⬇ Queue selected for embedding
              </button>
            )}
          </div>

          {/* Add source */}
          <div className="sb-section">
            <div className="sb-label">Add Source</div>
            <div style={{border:"1px solid var(--border)",borderRadius:8,overflow:"hidden"}}>
              <div className="tab-row" style={{background:"var(--panel)"}}>
                {[["pdf","📄 PDF"],["yt","▶ YouTube"],["web","🌐 Web"],["text","✏ Text"]].map(([id,lbl]) => (
                  <div key={id} className={`tab${ingestTab===id?" active":""}`} onClick={() => setIngestTab(id)}>{lbl}</div>
                ))}
              </div>
              <div className="tab-body">
                {ingestTab==="pdf" && (
                  <>
                    <input ref={fileRef} type="file" accept=".pdf,.txt,.csv,.png,.jpg,.mp4" style={{display:"none"}} onChange={onFile}/>
                    <button className="action-btn" style={{marginBottom:0,width:"100%"}} onClick={() => fileRef.current?.click()}>⬆ Choose file</button>
                  </>
                )}
                {ingestTab==="yt" && (
                  <div style={{display:"flex",gap:6}}>
                    <input className="sb-input" placeholder="YouTube URL…" value={ytUrl} onChange={e => setYtUrl(e.target.value)} style={{flex:1}}/>
                    <button className="send-btn" style={{width:32,height:32,borderRadius:8,fontSize:12}} onClick={addUrl}>+</button>
                  </div>
                )}
                {ingestTab==="web" && (
                  <div style={{display:"flex",gap:6}}>
                    <input className="sb-input" placeholder="https://…" value={webUrl} onChange={e => setWebUrl(e.target.value)} style={{flex:1}}/>
                    <button className="send-btn" style={{width:32,height:32,borderRadius:8,fontSize:12}} onClick={addUrl}>+</button>
                  </div>
                )}
                {ingestTab==="text" && (
                  <>
                    <textarea className="sb-input" rows={3} placeholder="Paste text here…"
                      value={pasteText} onChange={e => setPasteText(e.target.value)} style={{resize:"none"}}/>
                    <button className="action-btn" style={{marginTop:6,marginBottom:0,width:"100%"}} onClick={addPaste}>+ Add text</button>
                  </>
                )}
              </div>
            </div>

            <EmbedFlow
              pendingSources={pendingSources}
              chunker={chunker}    setChunker={setChunker}
              embModel={embModel}  setEmbModel={setEmbModel}
              embModels={embModels}
              onDone={handleEmbedDone}
              analysisData={analysisData}
              analyzing={analyzing}
            />
          </div>

          {/* Source list */}
          <div className="sb-section" style={{flex:1}}>
            <div className="sb-label" style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              Source List
              <span style={{color:"var(--muted)",fontFamily:"var(--font-mono)",fontSize:9,fontWeight:400}}>{sources.length} sources</span>
            </div>
            {sources.map(s => (
              <div key={s.id} className="source-row">
                <span className="src-icon">{SRC_ICONS[s.type]||"📁"}</span>
                <div className="src-meta">
                  <div style={{display:"flex",alignItems:"center",gap:4,marginBottom:2}}>
                    <span className="src-name" style={{flex:1}}>{s.name}</span>
                    <span className={`src-badge badge-${s.status}`}>{s.status}</span>
                  </div>
                  <div className="src-stats">
                    {s.chunks} chunks · {s.vectors} vecs · {embModels.find(e=>e.id===s.embModel)?.dim||"?"}d
                  </div>
                </div>
                <button className="src-del" onClick={() => deleteSource(s.id)} title="Remove">✕</button>
              </div>
            ))}
            <div className="totals-strip">
              <div className="total-cell"><span className="total-num">{totalSources}</span><span className="total-lbl">sources</span></div>
              <div className="total-cell"><span className="total-num">{totalChunks}</span> <span className="total-lbl">chunks</span></div>
              <div className="total-cell"><span className="total-num">{totalVectors}</span><span className="total-lbl">vectors</span></div>
            </div>
          </div>

          <div className="sb-section">
            <button className="new-chat-btn" onClick={newChat}>＋ New Chat</button>
          </div>
          <div className="sb-section" style={{flexShrink:0}}>
            <div className="sb-label">Chat History</div>
            {history.map(h => (
              <div key={h.id} className={`hist-item${activeSession===h.id?" active":""}`}
                onClick={() => loadSession(h.id)}>
                <div className={`hist-dot${activeSession===h.id?"":" old"}`}/>
                <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{h.title}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="sb-toggle"
          style={{left:sbOpen?"calc(var(--sb-w) - 1px)":"0px"}}
          onClick={() => setSbOpen(o => !o)}
          title={sbOpen?"Hide sidebar":"Show sidebar"}>
          {sbOpen?"‹":"›"}
        </div>

        {/* ── MAIN CHAT ── */}
        <div className="main">
          <div className="chat-header">
            <div className="header-title"><span>Mini</span>NotebookLM</div>
            <span className={`mode-badge ${modeBadge}`}>{modeLabel}</span>
            <div style={{flex:1}}/>
            <span style={{fontSize:10,color:"var(--muted)",fontFamily:"var(--font-mono)"}}>
              {totalSources} src · {totalChunks} chunks · {totalVectors} vecs
            </span>
          </div>

          <div className="messages">
            {messages.length===0 && (
              <div style={{margin:"auto",textAlign:"center",color:"var(--muted)",fontSize:13}}>
                <div style={{fontSize:36,marginBottom:10}}>✦</div>
                <div style={{fontWeight:700,color:"var(--text)",marginBottom:4}}>Ready to answer</div>
                <div>Add &amp; embed sources, then ask anything.</div>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={`msg${msg.role==="user"?" user":""}`}>
                <div className={`avatar ${msg.role==="user"?"av-user":"av-bot"}`}>
                  {msg.role==="user"?"U":"✦"}
                </div>
                <div className={`bubble ${msg.role==="user"?"bubble-user":"bubble-bot"}`}>
                  <div style={{whiteSpace:"pre-wrap"}}>
                    {msg.content}
                    {msg.citations?.map((c,i) => (
                      <span key={i} className="cite-tag" title={typeof c==="string"?c:(c.source||"")}>[{i+1}]</span>
                    ))}
                  </div>
                  {msg.chunks?.length>0 && (
                    <div className="chunks-section">
                      <div className="chunks-hdr"><span style={{color:"var(--accent)"}}>▪</span>Retrieved Chunks</div>
                      {msg.chunks.map((c,i) => (
                        <div key={i} className="chunk-item">
                          <div className="chunk-src">[{i+1}] {c.src || c.source || c.source_name || "source"}</div>
                          <div className="chunk-text">{c.text || c.content || ""}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="msg">
                <div className="avatar av-bot">✦</div>
                <div className="bubble bubble-bot">
                  <div className="loading-dots"><span/><span/><span/></div>
                </div>
              </div>
            )}
            <div ref={bottomRef}/>
          </div>

          <div className="input-bar">
            <textarea className="chat-input"
              placeholder={`Ask in ${modeLabel} mode… (Shift+Enter for newline)`}
              value={input} rows={1}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}/>
            <button className="send-btn" disabled={!input.trim()||loading} onClick={sendMessage}>↑</button>
          </div>

          <div className="statusbar">
            <span className="status-dot"/>
            <span>{provider==="groq" ? `Groq · ${model.slice(0,20)}` : `Ollama · ${ollamaUrl}`}</span>
            <span>·</span><span>{selEmb?.name}</span>
            <span>·</span><span>{selEmb?.dim}d</span>
            <span>·</span><span>{chunker}</span>
            <span>·</span><span>{stats.graph?.nodes||0} KG nodes</span>
          </div>
        </div>

        <div className="sb-toggle right"
          style={{right:rpOpen?"calc(var(--rp-w) - 1px)":"0px"}}
          onClick={() => setRpOpen(o => !o)}
          title={rpOpen?"Hide panel":"Show panel"}>
          {rpOpen?"›":"‹"}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div className={`right-panel${rpOpen?"":" collapsed"}`}>
          <div className="rp-section">
            <div className="rp-label">Knowledge Graph</div>
            <MiniGraph/>
            <div style={{fontSize:9,color:"var(--muted)",marginTop:6,fontFamily:"var(--font-mono)",textAlign:"center"}}>
              {stats.graph?.nodes||0} nodes · {stats.graph?.edges||0} edges
            </div>
          </div>

          <div className="rp-section">
            <div className="rp-label">Vector Store (multi-dim)</div>
            <div className="db-row"><span>Sources</span>      <span className="db-num">{totalSources}</span></div>
            <div className="db-row"><span>Total chunks</span> <span className="db-num">{totalChunks}</span></div>
            <div className="db-row"><span>Total vectors</span><span className="db-num">{totalVectors}</span></div>
            {dimBreakdown.length>0 && (
              <>
                <div style={{fontSize:9,color:"var(--muted)",margin:"8px 0 5px",letterSpacing:".1em",textTransform:"uppercase",fontWeight:700}}>By dimension</div>
                {dimBreakdown.map(e => (
                  <div key={e.dim} className="dim-row">
                    <span className="dim-label">{e.dim}d</span>
                    <div className="dim-bar" style={{width:`${Math.round(e.count/Math.max(totalVectors,1)*100)}%`}}/>
                    <span className="dim-count">{e.count}</span>
                  </div>
                ))}
              </>
            )}
          </div>

          <div className="rp-section">
            <div className="rp-label">Model Tuning</div>
            {[
              {label:"Temperature", key:"t", val:temp,      set:setTemp,      min:0,   max:2,    step:.05},
              {label:"Top-P",       key:"p", val:topP,      set:setTopP,      min:0,   max:1,    step:.05},
              {label:"Top-K",       key:"k", val:topK,      set:setTopK,      min:1,   max:100,  step:1},
              {label:"Max Tokens",  key:"m", val:maxTokens, set:setMaxTokens, min:128, max:4096, step:64},
            ].map(p => (
              <div key={p.key} className="param-row">
                <div className="param-label">{p.label}<span>{p.val}</span></div>
                <input type="range" min={p.min} max={p.max} step={p.step} value={p.val}
                  onChange={e => p.set(parseFloat(e.target.value))}/>
              </div>
            ))}
          </div>

          <div className="rp-section">
            <div className="rp-label">Actions</div>
            <button className="action-btn" onClick={() => { refreshStats(); showToast("Stats refreshed"); }}>↻ Refresh stats</button>
            <button className="action-btn" onClick={() => showToast("Cache cleared")}>🗑 Clear response cache</button>
            <button className="action-btn" onClick={() => showToast("Graph rebuilt")}>↻ Rebuild knowledge graph</button>
            <button className="action-btn danger"
              onClick={() => { setSources([]); setStats(s=>({...s,total_chunks:0,total_sources:0})); showToast("Vector store wiped"); }}>
              ⚠ Wipe vector store
            </button>
          </div>
        </div>

      </div>
      {toast && <div className={`toast${toastErr?" err":""}`}>{toast}</div>}
    </>
  );
}
