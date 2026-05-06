/**
 * Pure audio helpers — no MediaRecorder / encoder imports here so jsdom
 * tests can import this module without dragging in browser-only deps.
 */

const ACCEPTED_MIMES = new Set(["audio/wav", "audio/x-wav", "audio/wave"]);
export const MAX_AUDIO_BYTES = 50 * 1024 * 1024; // 50 MB cap (matches gateway)

export interface AudioValidation {
  ok: boolean;
  reason?: string;
}

export function validateAudioFile(file: File): AudioValidation {
  if (file.size > MAX_AUDIO_BYTES) {
    return {
      ok: false,
      reason: `Plik za duży (${(file.size / 1e6).toFixed(1)} MB). Maksymalny rozmiar: 50 MB.`,
    };
  }
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".wav")) return { ok: true };
  if (file.type && !ACCEPTED_MIMES.has(file.type)) {
    return {
      ok: false,
      reason: `Nieobsługiwany format (${file.type || "nieznany"}). Wymagany: WAV.`,
    };
  }
  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export function formatDuration(seconds: number): string {
  const mm = Math.floor(seconds / 60).toString().padStart(2, "0");
  const ss = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mm}:${ss}`;
}
