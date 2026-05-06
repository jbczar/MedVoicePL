"use client";

/**
 * MediaRecorder + WAV encoder bootstrap. Browser-only — keep in
 * `"use client"` files. Pure validation helpers live in
 * `audioValidation.ts`.
 */

import { register } from "extendable-media-recorder";
import { connect } from "extendable-media-recorder-wav-encoder";

let _wavEncoderRegistered = false;

/**
 * Register the WAV encoder with the global MediaRecorder shim. Idempotent.
 * Must be called once before constructing a `MediaRecorder({ mimeType: 'audio/wav' })`.
 *
 * Some browsers throw on second register — we swallow that and treat
 * "already registered" as success.
 */
export async function ensureWavEncoder(): Promise<void> {
  if (_wavEncoderRegistered) return;
  try {
    await register(await connect());
  } catch (err: unknown) {
    if (
      err instanceof Error &&
      /already registered/i.test(err.message ?? "")
    ) {
      _wavEncoderRegistered = true;
      return;
    }
    throw err;
  }
  _wavEncoderRegistered = true;
}

export {
  formatBytes,
  formatDuration,
  validateAudioFile,
  MAX_AUDIO_BYTES,
} from "./audioValidation";
export type { AudioValidation } from "./audioValidation";
