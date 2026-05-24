"use client";

import { useState } from "react";
import { ChevronDown, MessageSquare, User, UserCog } from "lucide-react";
import { cn } from "@/lib/cn";
import type { Transcript, TranscriptSegment } from "@/lib/types";

type Props = {
  transcript: Transcript;
};

export function TranscriptViewer({ transcript }: Props) {
  const [open, setOpen] = useState(false);
  const segments = transcript.segments ?? [];

  return (
    <section className="card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 rounded-md text-left transition hover:bg-slate-50"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <MessageSquare size={18} className="text-accent" aria-hidden />
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Transkrypcja rozmowy
            </h2>
            <p className="text-xs text-slate-500">
              {segments.length} wypowiedzi · {formatDuration(transcript.duration_s)} ·
              z STT (Whisper) + diaryzacja
            </p>
          </div>
        </div>
        <ChevronDown
          size={18}
          className={cn(
            "shrink-0 text-slate-400 transition-transform",
            open && "rotate-180"
          )}
          aria-hidden
        />
      </button>
      {open && (
        <div className="mt-4 space-y-2">
          {segments.length === 0 ? (
            <p className="text-sm text-slate-500">Brak segmentów do wyświetlenia.</p>
          ) : (
            segments.map((seg, i) => <SegmentRow key={i} segment={seg} />)
          )}
        </div>
      )}
    </section>
  );
}

function SegmentRow({ segment }: { segment: TranscriptSegment }) {
  const isDoctor = segment.speaker === "doctor";
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2",
        isDoctor
          ? "border-accent/20 bg-accent/5"
          : "border-slate-200 bg-slate-50"
      )}
    >
      <div className="flex items-center gap-2">
        {isDoctor ? (
          <UserCog
            size={14}
            className="text-accent"
            aria-label="Lekarz"
          />
        ) : (
          <User
            size={14}
            className="text-slate-500"
            aria-label="Pacjent"
          />
        )}
        <span
          className={cn(
            "text-xs font-medium",
            isDoctor ? "text-accent" : "text-slate-600"
          )}
        >
          {isDoctor ? "Lekarz" : segment.speaker === "patient" ? "Pacjent" : segment.speaker}
        </span>
        <span className="font-mono text-xs text-slate-400">
          {formatTime(segment.start_s)} – {formatTime(segment.end_s)}
        </span>
      </div>
      <p className="mt-1 text-sm leading-relaxed text-slate-700">{segment.text}</p>
    </div>
  );
}

function formatTime(s: number): string {
  const minutes = Math.floor(s / 60);
  const seconds = Math.floor(s % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatDuration(s: number): string {
  if (!s) return "0:00";
  return formatTime(s);
}
