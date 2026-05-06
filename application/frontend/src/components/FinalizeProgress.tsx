"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Loader2, Wand2, XCircle } from "lucide-react";
import { useEffect } from "react";

import * as api from "@/lib/api";
import type { FinalizeResponse, VisitStatus } from "@/lib/types";
import { cn } from "@/lib/cn";

const PHASE_LABELS: Record<string, string> = {
  audio_uploaded: "Gotowe do analizy",
  transcribing: "Transkrypcja audio",
  structuring: "Strukturyzacja SOAP",
  anonymizing: "Anonimizacja danych",
  suggesting: "Generowanie sugestii klinicznych",
  completed: "Ukończono",
  failed: "Nie powiodło się",
};

const PHASE_ORDER: VisitStatus[] = [
  "audio_uploaded",
  "transcribing",
  "anonymizing",
  "suggesting",
  "completed",
];

interface Props {
  visitId: string;
  onCompleted: (result: FinalizeResponse) => void;
}

/**
 * Triggers `POST /finalize` (which blocks ~30-180 s on cold start) AND
 * polls `GET /visits/{id}` every 1 s in parallel for the live status.
 *
 * The mutation's `onSuccess` carries the actual `FinalizeResponse` —
 * we don't need to refetch after. Polling exists purely so the doctor
 * sees the phase progress while the long fetch is in flight.
 */
export function FinalizeProgress({ visitId, onCompleted }: Props) {
  const qc = useQueryClient();
  const finalize = useMutation({
    mutationFn: () => api.finalizeVisit(visitId),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["visit", visitId] });
      onCompleted(data);
    },
  });

  const polling = useQuery({
    queryKey: ["visit", visitId],
    queryFn: () => api.getVisit(visitId),
    refetchInterval: finalize.isPending ? 1000 : false,
    enabled: finalize.isPending,
  });

  useEffect(() => {
    // Auto-start finalize when this component mounts (visit is in audio_uploaded).
    if (
      polling.data?.status === "audio_uploaded" &&
      !finalize.isPending &&
      !finalize.isSuccess &&
      !finalize.isError
    ) {
      finalize.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polling.data?.status]);

  const currentStatus = polling.data?.status ?? "audio_uploaded";
  const isFailed = finalize.isError || currentStatus === "failed";
  const isDone = finalize.isSuccess;

  return (
    <div className="card p-6">
      <div className="mb-5 flex items-start gap-3">
        <div
          className={cn(
            "rounded-lg p-2",
            isFailed
              ? "bg-red-50 text-danger"
              : isDone
                ? "bg-emerald-50 text-emerald-600"
                : "bg-accent-subtle text-accent"
          )}
        >
          {isFailed ? (
            <XCircle className="h-5 w-5" />
          ) : isDone ? (
            <CheckCircle2 className="h-5 w-5" />
          ) : (
            <Wand2 className="h-5 w-5" />
          )}
        </div>
        <div>
          <h2 className="text-lg font-semibold">
            {isDone
              ? "Analiza ukończona"
              : isFailed
                ? "Analiza nie powiodła się"
                : "Analizujemy wizytę"}
          </h2>
          <p className="mt-0.5 text-sm text-slate-500">
            {isDone
              ? "Sugestie kliniczne są gotowe poniżej."
              : isFailed
                ? "Spróbuj ponownie lub skontaktuj się z administratorem."
                : "Pierwsze uruchomienie może potrwać do kilku minut (ładowanie modeli)."}
          </p>
        </div>
      </div>

      <ol className="space-y-2">
        {PHASE_ORDER.map((phase, idx) => {
          const phaseIdx = PHASE_ORDER.indexOf(currentStatus);
          const reached = phaseIdx >= idx;
          const active = currentStatus === phase && !isDone && !isFailed;
          const wasFailed = isFailed && idx === phaseIdx;
          return (
            <li
              key={phase}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm",
                active && "bg-accent-subtle",
                wasFailed && "bg-red-50"
              )}
            >
              <span
                className={cn(
                  "flex h-5 w-5 items-center justify-center rounded-full",
                  reached && !isFailed ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-400",
                  active && "bg-accent-subtle text-accent",
                  wasFailed && "bg-red-100 text-danger"
                )}
                aria-hidden
              >
                {active ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : reached && !isFailed ? (
                  <CheckCircle2 className="h-3 w-3" />
                ) : wasFailed ? (
                  <XCircle className="h-3 w-3" />
                ) : (
                  <span className="text-[10px]">{idx + 1}</span>
                )}
              </span>
              <span
                className={cn(
                  reached ? "font-medium text-slate-800" : "text-slate-500",
                  active && "text-accent"
                )}
              >
                {PHASE_LABELS[phase] ?? phase}
              </span>
            </li>
          );
        })}
      </ol>

      {finalize.isError && (
        <div className="mt-5 flex items-center justify-between gap-3 rounded-md border border-danger/30 bg-red-50 px-3 py-2 text-sm text-danger">
          <span>
            {finalize.error instanceof Error
              ? finalize.error.message
              : "Wystąpił błąd."}
          </span>
          <button
            type="button"
            className="btn-secondary text-xs"
            onClick={() => finalize.mutate()}
          >
            Spróbuj ponownie
          </button>
        </div>
      )}
    </div>
  );
}
