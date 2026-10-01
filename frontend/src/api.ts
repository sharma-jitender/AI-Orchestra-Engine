export const API_BASE = import.meta.env.VITE_API_URL || "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export type Metrics = {
  requests: number;
  total_cost: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  avg_quality: number;
  success_rate: number;
  fallback_rate: number;
};

export type Overview = {
  routed: Metrics;
  baseline: Metrics;
  cost_savings_vs_baseline: number | null;
};

export type CompleteResponse = {
  output: string;
  task_type: string;
  model: string;
  provider: string;
  policy: string;
  tokens_in: number;
  tokens_out: number;
  est_cost: number;
  latency_ms: number;
  success: boolean;
  fallback_used: boolean;
  fallback_from: string | null;
  quality_score: number;
  ranked: string[];
};

export const api = {
  overview: () => req<Overview>("/v1/stats/overview"),
  byModel: () => req<Array<Record<string, string | number>>>("/v1/stats/by-model"),
  byTask: () => req<Array<Record<string, string | number>>>("/v1/stats/by-task"),
  requests: () => req<Array<Record<string, unknown>>>("/v1/requests?limit=80"),
  models: () => req<{ baseline_model: string; weights: Record<string, number>; models: Array<Record<string, unknown>> }>("/v1/models"),
  latestEval: () => req<Overview & { id: string | null; prompt_count?: number }>("/v1/eval/latest"),
  runEval: (limit?: number) =>
    req<Overview & { id: string; prompt_count: number }>("/v1/eval/run", {
      method: "POST",
      body: JSON.stringify({ limit: limit ?? null }),
    }),
  complete: (prompt: string, policy: "routed" | "baseline") =>
    req<CompleteResponse>("/v1/complete", {
      method: "POST",
      body: JSON.stringify({ prompt, policy }),
    }),
};
