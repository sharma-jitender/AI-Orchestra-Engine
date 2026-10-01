import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type CompleteResponse, type Metrics, type Overview } from "./api";

type Tab = "overview" | "baseline" | "playground" | "requests" | "registry";

const navItems: Array<{ id: Tab; label: string; icon: string }> = [
  { id: "overview", label: "Overview", icon: "◫" },
  { id: "baseline", label: "Baseline", icon: "↗" },
  { id: "playground", label: "Playground", icon: "✳" },
  { id: "requests", label: "Requests", icon: "≡" },
  { id: "registry", label: "Registry", icon: "▦" },
];

function fmt(n: number, d = 4) {
  return n.toFixed(d);
}
function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

function MetricCards({ title, m, savings }: { title: string; m: Metrics; savings?: number | null }) {
  return (
    <section className="metrics-block">
      <h2 className="section-title">{title}</h2>
      <div className="metric-grid">
        <div className="metric-card">
          <h3>Requests</h3>
          <div className="metric-value">{m.requests}</div>
        </div>
        <div className="metric-card">
          <h3>Total cost</h3>
          <div className="metric-value">${fmt(m.total_cost, 5)}</div>
          {savings != null && <div className="metric-note">{pct(savings)} cheaper than baseline</div>}
        </div>
        <div className="metric-card">
          <h3>Avg latency</h3>
          <div className="metric-value">{m.avg_latency_ms} <small>ms</small></div>
        </div>
        <div className="metric-card">
          <h3>Quality / success</h3>
          <div className="metric-value">{fmt(m.avg_quality, 3)}</div>
          <div className="metric-detail">Success {pct(m.success_rate)} <span>·</span> Fallback {pct(m.fallback_rate)}</div>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [byModel, setByModel] = useState<Array<Record<string, string | number>>>([]);
  const [byTask, setByTask] = useState<Array<Record<string, string | number>>>([]);
  const [requests, setRequests] = useState<Array<Record<string, unknown>>>([]);
  const [models, setModels] = useState<{ baseline_model: string; weights: Record<string, number>; models: Array<Record<string, unknown>> } | null>(null);
  const [evalResult, setEvalResult] = useState<(Overview & { id: string | null; prompt_count?: number }) | null>(null);
  const [prompt, setPrompt] = useState("Write a Python function that reverses a linked list.");
  const [completion, setCompletion] = useState<CompleteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setError(null);
    try {
      const [o, m, t, r, md, ev] = await Promise.all([
        api.overview(),
        api.byModel(),
        api.byTask(),
        api.requests(),
        api.models(),
        api.latestEval(),
      ]);
      setOverview(o);
      setByModel(m);
      setByTask(t);
      setRequests(r);
      setModels(md);
      setEvalResult(ev.id ? ev : null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  const taskChart = useMemo(() => {
    const map = new Map<string, { task: string; routed: number; baseline: number }>();
    for (const row of byTask) {
      const task = String(row.task_type);
      const cur = map.get(task) || { task, routed: 0, baseline: 0 };
      if (row.policy === "routed") cur.routed = Number(row.total_cost);
      else cur.baseline = Number(row.total_cost);
      map.set(task, cur);
    }
    return [...map.values()];
  }, [byTask]);

  async function onComplete(policy: "routed" | "baseline") {
    setBusy(true);
    setError(null);
    try {
      const res = await api.complete(prompt, policy);
      setCompletion(res);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "complete failed");
    } finally {
      setBusy(false);
    }
  }

  async function onEval() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.runEval();
      setEvalResult(res);
      setTab("baseline");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "eval failed");
    } finally {
      setBusy(false);
    }
  }

  const pageTitles: Record<Tab, string> = {
    overview: "Overview",
    baseline: "Baseline comparison",
    playground: "Playground",
    requests: "Requests",
    registry: "Model registry",
  };

  return (
    <div className={`app-shell${sidebarCollapsed ? " sidebar-collapsed" : ""}`} data-theme={darkMode ? "dark" : "light"}>
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          {!sidebarCollapsed && <span className="brand-name">Orchestrator</span>}
          <button
            className="icon-button collapse-button"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? "›" : "‹"}
          </button>
        </div>
        <nav className="side-nav" aria-label="Main navigation">
          <span className="nav-caption">Workspace</span>
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item${tab === item.id ? " active" : ""}`}
              onClick={() => setTab(item.id)}
              title={sidebarCollapsed ? item.label : undefined}
              aria-current={tab === item.id ? "page" : undefined}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              {!sidebarCollapsed && <span>{item.label}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)} title={darkMode ? "Switch to light theme" : "Switch to dark theme"}>
            <span className="theme-icon" aria-hidden="true">{darkMode ? "☼" : "◐"}</span>
            {!sidebarCollapsed && <span>{darkMode ? "Light appearance" : "Dark appearance"}</span>}
          </button>
          {!sidebarCollapsed && <div className="sidebar-footnote">Local orchestration workspace</div>}
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="page-heading">
            <span className="breadcrumb">Workspace <span>/</span> {pageTitles[tab]}</span>
            <h1>{pageTitles[tab]}</h1>
          </div>
          <div className="topbar-actions">
            <span className="status-indicator"><i /> System ready</span>
            <button className="primary-button" disabled={busy} onClick={() => void onEval()}>
              <span aria-hidden="true">↻</span> {busy ? "Working…" : "Run baseline eval"}
            </button>
          </div>
        </header>

        <div className="content-area">
          {error && <p className="error-banner" role="alert">{error}</p>}

          {tab === "overview" && overview && (
            <div className="page-stack">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">PERFORMANCE</p>
                  <h2>Routing at a glance</h2>
                  <p className="section-description">Compare the routed policy with your locked single-model baseline.</p>
                </div>
                <span className="subtle-label">All logged requests</span>
              </div>
              <div className="comparison-grid">
                <MetricCards title="Routed policy" m={overview.routed} savings={overview.cost_savings_vs_baseline} />
                <MetricCards title="Baseline policy" m={overview.baseline} />
              </div>
              <div className="dashboard-grid">
                <section className="panel chart-panel">
                  <div className="panel-heading">
                    <div><h3>Cost by task</h3><p>Routed and baseline spend, grouped by request type</p></div>
                    <span className="panel-menu" aria-hidden="true">···</span>
                  </div>
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={taskChart} margin={{ top: 12, right: 8, left: -14, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="2 6" stroke="var(--line)" vertical={false} />
                        <XAxis dataKey="task" axisLine={false} tickLine={false} tick={{ fill: "var(--muted)", fontSize: 12 }} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }} />
                        <Tooltip cursor={{ fill: "var(--hover)" }} contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8, color: "var(--text)" }} />
                        <Legend iconType="circle" wrapperStyle={{ color: "var(--muted)", fontSize: 12 }} />
                        <Bar dataKey="routed" name="Routed" fill="#1783FF" radius={[4, 4, 0, 0]} maxBarSize={30} />
                        <Bar dataKey="baseline" name="Baseline" fill="var(--chart-neutral)" radius={[4, 4, 0, 0]} maxBarSize={30} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
                <section className="panel table-panel">
                  <div className="panel-heading">
                    <div><h3>Traffic by model</h3><p>Usage across the current workspace</p></div>
                    <span className="panel-menu" aria-hidden="true">···</span>
                  </div>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead><tr><th>Model</th><th>Requests</th><th>Cost</th><th>Latency</th></tr></thead>
                      <tbody>{byModel.map((r) => (
                        <tr key={String(r.model)}><td className="strong-cell">{String(r.model)}</td><td>{String(r.requests)}</td><td>${fmt(Number(r.total_cost), 5)}</td><td>{String(r.avg_latency_ms)} ms</td></tr>
                      ))}</tbody>
                    </table>
                  </div>
                </section>
              </div>
            </div>
          )}

          {tab === "baseline" && (
            <div className="page-stack">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">EVALUATION</p>
                  <h2>Locked-model comparison</h2>
                  <p className="section-description">Same fixture prompts, evaluated by the router and {models?.baseline_model ?? "the baseline model"}.</p>
                </div>
                {evalResult && <span className="subtle-label">{evalResult.prompt_count} prompts</span>}
              </div>
              {!evalResult && <div className="empty-panel"><span className="empty-icon">↗</span><h3>No evaluation yet</h3><p>Run the baseline evaluation to compare cost, latency, and quality.</p></div>}
              {evalResult?.routed && (
                <>
                  <div className="run-summary"><span>Latest run</span><code>{evalResult.id}</code></div>
                  <div className="comparison-grid">
                    <MetricCards title="Routed policy" m={evalResult.routed} savings={evalResult.cost_savings_vs_baseline} />
                    <MetricCards title="Baseline policy" m={evalResult.baseline} />
                  </div>
                </>
              )}
            </div>
          )}

          {tab === "playground" && (
            <section className="playground-view">
              <div className="chat-scroll">
                {!completion ? (
                  <div className="welcome-block">
                    <div className="welcome-mark"><span /></div>
                    <p className="eyebrow">MODEL PLAYGROUND</p>
                    <h2>What would you like to route?</h2>
                    <p>Send a prompt through the scoring router or compare it with your baseline model.</p>
                  </div>
                ) : (
                  <div className="conversation">
                    <div className="message-row user-message"><div className="message-bubble">{prompt}</div></div>
                    <div className="assistant-message">
                      <div className="assistant-avatar"><span /></div>
                      <div className="assistant-content">
                        <div className="assistant-name">Orchestrator</div>
                        <pre className="assistant-output">{completion.output}</pre>
                        <div className="message-chips">
                          <span className="meta-chip model-chip">{completion.model}</span>
                          <span className="meta-chip">${fmt(completion.est_cost, 5)}</span>
                          <span className="meta-chip">{completion.latency_ms} ms</span>
                          <span className="meta-chip">Quality {fmt(completion.quality_score, 3)}</span>
                          {completion.fallback_used && <span className="meta-chip">Fallback from {completion.fallback_from}</span>}
                        </div>
                        <p className="ranked-line">{completion.task_type} <span>·</span> ranked {completion.ranked.join(" → ")}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div className="composer-dock">
                <div className="chat-composer">
                  <textarea aria-label="Your prompt" placeholder="Ask anything or describe a task..." value={prompt} onChange={(e) => setPrompt(e.target.value)} />
                  <div className="composer-toolbar">
                    <span className="composer-hint">Responses use the configured routing policy</span>
                    <div className="composer-actions">
                      <button className="secondary-button" disabled={busy} onClick={() => void onComplete("baseline")}>Baseline</button>
                      <button className="send-button" disabled={busy || !prompt.trim()} onClick={() => void onComplete("routed")} aria-label="Route prompt">
                        {busy ? "…" : "↑"}
                      </button>
                    </div>
                  </div>
                </div>
                <p className="composer-disclaimer">Mock adapters are active when provider credentials are unavailable.</p>
              </div>
            </section>
          )}

          {tab === "requests" && (
            <section className="panel full-panel">
              <div className="panel-heading">
                <div><h3>Recent requests</h3><p>Routing decisions and measured outcomes</p></div>
                <span className="subtle-label">{requests.length} records</span>
              </div>
              <div className="table-wrap">
                <table className="data-table requests-table">
                  <thead><tr><th>ID</th><th>Task</th><th>Policy</th><th>Model</th><th>Cost</th><th>Latency</th><th>Quality</th><th>Prompt</th></tr></thead>
                  <tbody>{requests.map((r) => (
                    <tr key={String(r.id)}><td>{String(r.id)}</td><td><span className="type-label">{String(r.task_type)}</span></td><td>{String(r.policy)}</td><td className="strong-cell">{String(r.model)}{r.fallback_used ? " *" : ""}</td><td>${fmt(Number(r.est_cost), 5)}</td><td>{String(r.latency_ms)} ms</td><td>{fmt(Number(r.quality_score), 2)}</td><td className="prompt-cell">{String(r.prompt)}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </section>
          )}

          {tab === "registry" && models && (
            <section className="panel full-panel">
              <div className="panel-heading">
                <div><h3>Available models</h3><p>Relative costs, latency, and task strengths</p></div>
                <span className="subtle-label">Baseline · {models.baseline_model}</span>
              </div>
              <div className="registry-summary">Routing weights <strong>Quality {models.weights.quality}</strong><strong>Cost {models.weights.cost}</strong><strong>Latency {models.weights.latency}</strong></div>
              <div className="table-wrap">
                <table className="data-table">
                  <thead><tr><th>Model</th><th>Provider</th><th>Input / 1k</th><th>Latency</th><th>Code</th><th>Summary</th><th>Reasoning</th></tr></thead>
                  <tbody>{models.models.map((m) => {
                    const s = m.strengths as Record<string, number>;
                    return <tr key={String(m.id)}><td className="strong-cell">{String(m.display_name)}</td><td>{String(m.provider)}</td><td>${String(m.input_cost_per_1k)}</td><td>{String(m.typical_latency_ms)} ms</td><td>{s.code}</td><td>{s.summarization}</td><td>{s.reasoning}</td></tr>;
                  })}</tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
