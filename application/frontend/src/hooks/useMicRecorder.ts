"use client";

import { MediaRecorder, IMediaRecorder, type IMediaRecorderEventMap } from "extendable-media-recorder";
import { useCallback, useEffect, useRef, useState } from "react";

import { ensureWavEncoder } from "@/lib/audio";

type RecorderState = "idle" | "starting" | "recording" | "stopped" | "error";

interface UseMicRecorderResult {
  state: RecorderState;
  error: string | null;
  blob: Blob | null;
  durationS: number;
  /** 0..1 instantaneous RMS level for the meter UI. */
  level: number;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  reset: () => void;
}

/**
 * MediaRecorder wrapper producing WAV (mono) via the registered encoder.
 *
 * Stream lifecycle: getUserMedia → MediaRecorder → ondataavailable
 * → on stop emit a single Blob. We also tap an AnalyserNode for the
 * RMS level meter (purely cosmetic, but it gives the doctor confidence
 * the mic is picking up audio).
 */
export function useMicRecorder(): UseMicRecorderResult {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [durationS, setDurationS] = useState(0);
  const [level, setLevel] = useState(0);

  const recorderRef = useRef<IMediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const startedAtRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const cleanup = useCallback(() => {
    if (tickRef.current !== null) {
      cancelAnimationFrame(tickRef.current);
      tickRef.current = null;
    }
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      try {
        recorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    recorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => undefined);
      audioCtxRef.current = null;
    }
    analyserRef.current = null;
  }, []);

  useEffect(() => () => cleanup(), [cleanup]);

  const tickMeter = useCallback(() => {
    const a = analyserRef.current;
    if (!a) return;
    const buf = new Uint8Array(a.fftSize);
    a.getByteTimeDomainData(buf);
    let sumSq = 0;
    for (const sample of buf) {
      const norm = (sample - 128) / 128;
      sumSq += norm * norm;
    }
    const rms = Math.sqrt(sumSq / buf.length);
    setLevel(Math.min(1, rms * 3)); // amplify for visual range
    if (startedAtRef.current !== null) {
      setDurationS((Date.now() - startedAtRef.current) / 1000);
    }
    tickRef.current = requestAnimationFrame(tickMeter);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setBlob(null);
    setDurationS(0);
    setLevel(0);
    setState("starting");
    chunksRef.current = [];

    try {
      await ensureWavEncoder();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // Analyser tap for the level meter.
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyserRef.current = analyser;

      const recorder = new MediaRecorder(stream, { mimeType: "audio/wav" });
      recorderRef.current = recorder;
      recorder.addEventListener("dataavailable", (event: IMediaRecorderEventMap["dataavailable"]) => {
        if (event.data && event.data.size > 0) chunksRef.current.push(event.data);
      });
      recorder.addEventListener("stop", () => {
        const wav = new Blob(chunksRef.current, { type: "audio/wav" });
        setBlob(wav);
        setState("stopped");
        if (tickRef.current !== null) {
          cancelAnimationFrame(tickRef.current);
          tickRef.current = null;
        }
        setLevel(0);
      });
      recorder.start();
      startedAtRef.current = Date.now();
      tickRef.current = requestAnimationFrame(tickMeter);
      setState("recording");
    } catch (err: unknown) {
      cleanup();
      setState("error");
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError(
          "Brak dostępu do mikrofonu. Sprawdź uprawnienia w ustawieniach przeglądarki."
        );
      } else if (err instanceof DOMException && err.name === "NotFoundError") {
        setError("Nie wykryto mikrofonu.");
      } else {
        setError(
          err instanceof Error ? err.message : "Nie udało się uruchomić nagrywania."
        );
      }
    }
  }, [cleanup, tickMeter]);

  const stop = useCallback(async () => {
    const r = recorderRef.current;
    if (!r) return;
    if (r.state === "inactive") return;
    r.stop();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
  }, []);

  const reset = useCallback(() => {
    cleanup();
    setState("idle");
    setError(null);
    setBlob(null);
    setDurationS(0);
    setLevel(0);
    chunksRef.current = [];
    startedAtRef.current = null;
  }, [cleanup]);

  return { state, error, blob, durationS, level, start, stop, reset };
}
