import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";

const originalFetch = globalThis.fetch;

function mockFetch(impl: typeof fetch) {
  globalThis.fetch = vi.fn(impl) as unknown as typeof fetch;
}

beforeEach(() => {
  globalThis.fetch = originalFetch;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api.login", () => {
  it("sends credentials and returns void on 204", async () => {
    mockFetch(async (url, init) => {
      expect(String(url)).toMatch(/\/v1\/auth\/login$/);
      expect(init?.credentials).toBe("include");
      const headers = new Headers(init?.headers);
      expect(headers.get("Content-Type")).toBe("application/json");
      const body = JSON.parse(String(init?.body));
      expect(body).toEqual({ username: "u", password: "p" });
      return new Response(null, { status: 204 });
    });
    await expect(api.login("u", "p")).resolves.toBeUndefined();
  });

  it("throws ApiError with status on 401", async () => {
    mockFetch(
      async () =>
        new Response(JSON.stringify({ detail: "invalid_credentials" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        })
    );
    await expect(api.login("u", "p")).rejects.toMatchObject({
      status: 401,
    });
    await expect(api.login("u", "p")).rejects.toBeInstanceOf(ApiError);
  });
});

describe("api.uploadAudio", () => {
  it("sends multipart with the audio blob", async () => {
    mockFetch(async (url, init) => {
      expect(String(url)).toMatch(/\/v1\/visits\/abc\/audio$/);
      expect(init?.method).toBe("POST");
      expect(init?.body).toBeInstanceOf(FormData);
      const form = init?.body as FormData;
      expect(form.get("file")).toBeInstanceOf(Blob);
      return new Response(
        JSON.stringify({
          visit_id: "abc",
          status: "audio_uploaded",
          audio_sha256: "x".repeat(64),
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    });
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" });
    const out = await api.uploadAudio("abc", blob, "test.wav");
    expect(out.status).toBe("audio_uploaded");
  });
});

describe("api.health", () => {
  it("parses HealthResponse", async () => {
    mockFetch(
      async () =>
        new Response(
          JSON.stringify({
            status: "ok",
            on_premise_reachable: true,
            cloud_reachable: true,
            version: "0.1.0",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
    );
    const h = await api.health();
    expect(h.status).toBe("ok");
    expect(h.on_premise_reachable).toBe(true);
  });
});
