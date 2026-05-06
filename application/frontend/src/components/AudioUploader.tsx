"use client";

import { useMutation } from "@tanstack/react-query";
import { Check, FileAudio, Mic, MicOff, Square, Upload, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { useMicRecorder } from "@/hooks/useMicRecorder";
import * as api from "@/lib/api";
import {
  formatBytes,
  formatDuration,
  validateAudioFile,
} from "@/lib/audio";
import { cn } from "@/lib/cn";

type Mode = "file" | "mic";

interface AudioUploaderProps {
  visitId: string;
  onUploaded: () => void;
}

export function AudioUploader({ visitId, onUploaded }: AudioUploaderProps) {
  const [mode, setMode] = useState<Mode>("file");
  return (
    <div className="card overflow-hidden">
      <div className="flex border-b border-slate-200">
        <ModeTab
          active={mode === "file"}
          onClick={() => setMode("file")}
          icon={<FileAudio className="h-4 w-4" />}
          label="Wybierz plik"
        />
        <ModeTab
          active={mode === "mic"}
          onClick={() => setMode("mic")}
          icon={<Mic className="h-4 w-4" />}
          label="Nagraj mikrofonem"
        />
      </div>
      <div className="p-6">
        {mode === "file" ? (
          <FileMode visitId={visitId} onUploaded={onUploaded} />
        ) : (
          <MicMode visitId={visitId} onUploaded={onUploaded} />
        )}
      </div>
    </div>
  );
}

function ModeTab({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-1 items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors",
        active
          ? "border-b-2 border-accent bg-accent-subtle/40 text-accent"
          : "border-b-2 border-transparent text-slate-500 hover:bg-slate-50"
      )}
      aria-pressed={active}
    >
      {icon}
      {label}
    </button>
  );
}

// --- Shared upload mutation ---

function useUploadMutation(visitId: string, onUploaded: () => void) {
  return useMutation({
    mutationFn: ({ blob, name }: { blob: Blob; name: string }) =>
      api.uploadAudio(visitId, blob, name),
    onSuccess: () => onUploaded(),
  });
}

// --- File mode ---

function FileMode({ visitId, onUploaded }: AudioUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const upload = useUploadMutation(visitId, onUploaded);

  function pick(picked: File | null) {
    setError(null);
    if (!picked) {
      setFile(null);
      return;
    }
    const validation = validateAudioFile(picked);
    if (!validation.ok) {
      setError(validation.reason ?? "Nieprawidłowy plik.");
      setFile(null);
      return;
    }
    setFile(picked);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    pick(e.dataTransfer.files[0] ?? null);
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors",
          dragOver
            ? "border-accent bg-accent-subtle/40"
            : "border-slate-300 hover:border-slate-400 hover:bg-slate-50"
        )}
        role="button"
        tabIndex={0}
      >
        <Upload className="h-6 w-6 text-slate-400" />
        <div className="text-sm font-medium text-slate-700">
          Przeciągnij plik WAV lub kliknij, aby wybrać
        </div>
        <div className="text-xs text-slate-500">Maks. 50 MB · format WAV</div>
        <input
          ref={inputRef}
          type="file"
          accept="audio/wav,.wav"
          className="hidden"
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
      </div>

      {file && (
        <div className="flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
          <div className="flex items-center gap-2 text-slate-700">
            <FileAudio className="h-4 w-4 text-slate-500" />
            <span className="truncate" title={file.name}>
              {file.name}
            </span>
            <span className="text-xs text-slate-500">{formatBytes(file.size)}</span>
          </div>
          <button
            type="button"
            className="text-slate-400 hover:text-slate-600"
            onClick={() => setFile(null)}
            aria-label="Usuń plik"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {error && (
        <div role="alert" className="rounded-md border border-danger/30 bg-red-50 px-3 py-2 text-sm text-danger">
          {error}
        </div>
      )}

      {upload.isError && (
        <div role="alert" className="rounded-md border border-danger/30 bg-red-50 px-3 py-2 text-sm text-danger">
          Nie udało się przesłać pliku. Spróbuj ponownie.
        </div>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          className="btn-primary"
          disabled={!file || upload.isPending}
          onClick={() => file && upload.mutate({ blob: file, name: file.name })}
        >
          {upload.isPending ? "Przesyłanie…" : (
            <>
              <Upload className="h-4 w-4" />
              Prześlij
            </>
          )}
        </button>
      </div>
    </div>
  );
}

// --- Mic mode ---

function MicMode({ visitId, onUploaded }: AudioUploaderProps) {
  const recorder = useMicRecorder();
  const upload = useUploadMutation(visitId, onUploaded);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!recorder.blob) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(recorder.blob);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [recorder.blob]);

  const isRecording = recorder.state === "recording";
  const isStopped = recorder.state === "stopped" && !!recorder.blob;

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center gap-4 rounded-lg border border-slate-200 bg-slate-50 px-6 py-8">
        <div
          className={cn(
            "relative flex h-20 w-20 items-center justify-center rounded-full transition-colors",
            isRecording ? "bg-red-50" : "bg-white shadow-card"
          )}
          aria-hidden
        >
          {isRecording && (
            <span
              className="absolute inset-0 rounded-full border-2 border-danger/30"
              style={{
                transform: `scale(${1 + recorder.level * 0.4})`,
                transition: "transform 60ms linear",
              }}
            />
          )}
          {isRecording ? (
            <Mic className="h-8 w-8 text-danger" />
          ) : isStopped ? (
            <Check className="h-8 w-8 text-emerald-600" />
          ) : (
            <MicOff className="h-8 w-8 text-slate-400" />
          )}
        </div>

        <div className="font-mono text-2xl font-semibold tabular-nums text-slate-800">
          {formatDuration(recorder.durationS)}
        </div>

        {recorder.error && (
          <div role="alert" className="text-sm text-danger">
            {recorder.error}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-center gap-2">
          {!isRecording && !isStopped && (
            <button
              type="button"
              className="btn-primary"
              onClick={() => recorder.start()}
              disabled={recorder.state === "starting"}
            >
              <Mic className="h-4 w-4" />
              Rozpocznij nagrywanie
            </button>
          )}
          {isRecording && (
            <button
              type="button"
              className="btn-danger"
              onClick={() => recorder.stop()}
            >
              <Square className="h-4 w-4" />
              Zatrzymaj
            </button>
          )}
          {isStopped && (
            <>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => recorder.reset()}
              >
                <X className="h-4 w-4" />
                Odrzuć
              </button>
              <button
                type="button"
                className="btn-primary"
                disabled={!recorder.blob || upload.isPending}
                onClick={() =>
                  recorder.blob &&
                  upload.mutate({
                    blob: recorder.blob,
                    name: `recording-${new Date().toISOString().slice(0, 16)}.wav`,
                  })
                }
              >
                {upload.isPending ? "Przesyłanie…" : (
                  <>
                    <Upload className="h-4 w-4" />
                    Prześlij
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>

      {previewUrl && (
        <div>
          <div className="mb-2 text-sm font-medium text-slate-700">Odsłuchaj nagranie</div>
          <audio src={previewUrl} controls className="w-full" />
          {recorder.blob && (
            <div className="mt-1 text-xs text-slate-500">
              Rozmiar: {formatBytes(recorder.blob.size)} · WAV mono
            </div>
          )}
        </div>
      )}

      {upload.isError && (
        <div role="alert" className="rounded-md border border-danger/30 bg-red-50 px-3 py-2 text-sm text-danger">
          Nie udało się przesłać nagrania. Spróbuj ponownie.
        </div>
      )}
    </div>
  );
}
