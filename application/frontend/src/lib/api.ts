/**
 * Thin fetch wrapper over the gateway HTTP API (Phase 3b).
 *
 * Every request:
 *   - sends `credentials: "include"` (httpOnly session cookie)
 *   - throws `ApiError` on non-2xx responses (preserves status + detail)
 *
 * The base URL comes from `NEXT_PUBLIC_GATEWAY_URL` (default
 * `http://127.0.0.1:8080`). React Query layers caching/retries on top.
 */

import type {
  AgentSuggestion,
  AudioUploadResponse,
  CreateVisitResponse,
  DoctorIdentity,
  FinalizeResponse,
  HealthResponse,
  Specialization,
  VisitSummary,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? `API error ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

const BASE_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://127.0.0.1:8080";

async function request<T>(
  path: string,
  init: RequestInit & { json?: unknown } = {}
): Promise<T> {
  const { json, headers, ...rest } = init;
  const opts: RequestInit = {
    ...rest,
    credentials: "include",
    headers: {
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(headers ?? {}),
    },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  };
  const url = `${BASE_URL}${path}`;
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    let detail: unknown = null;
    try {
      detail = await resp.json();
    } catch {
      detail = await resp.text();
    }
    throw new ApiError(resp.status, detail);
  }
  // 204 No Content / etc.
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

// --- Auth ---

export async function login(username: string, password: string): Promise<void> {
  await request<void>("/v1/auth/login", {
    method: "POST",
    json: { username, password },
  });
}

export async function logout(): Promise<void> {
  await request<void>("/v1/auth/logout", { method: "POST" });
}

export async function me(): Promise<DoctorIdentity> {
  return request<DoctorIdentity>("/v1/auth/me");
}

// --- Visits ---

export async function createVisit(input: {
  specialization?: Specialization;
  visit_date?: string;
}): Promise<CreateVisitResponse> {
  return request<CreateVisitResponse>("/v1/visits", {
    method: "POST",
    json: input,
  });
}

export async function getVisit(visitId: string): Promise<VisitSummary> {
  return request<VisitSummary>(`/v1/visits/${visitId}`);
}

export async function uploadAudio(
  visitId: string,
  audio: Blob,
  filename = "audio.wav"
): Promise<AudioUploadResponse> {
  const form = new FormData();
  form.append("file", audio, filename);
  return request<AudioUploadResponse>(`/v1/visits/${visitId}/audio`, {
    method: "POST",
    body: form,
  });
}

export async function finalizeVisit(visitId: string): Promise<FinalizeResponse> {
  return request<FinalizeResponse>(`/v1/visits/${visitId}/finalize`, {
    method: "POST",
  });
}

// --- Health ---

export async function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/v1/health");
}

// Re-export types for convenience.
export type {
  AgentSuggestion,
  AudioUploadResponse,
  CreateVisitResponse,
  DoctorIdentity,
  FinalizeResponse,
  HealthResponse,
  Specialization,
  VisitSummary,
};
