"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { AudioUploader } from "@/components/AudioUploader";
import { AuthGuard } from "@/components/AuthGuard";
import { FinalizeProgress } from "@/components/FinalizeProgress";
import { Header } from "@/components/Header";
import { SuggestionView } from "@/components/SuggestionView";
import * as api from "@/lib/api";
import type { FinalizeResponse } from "@/lib/types";
import { useVisitStore } from "@/lib/visitStore";

const SPECIALIZATION_LABELS: Record<string, string> = {
  kardiologia: "Kardiologia",
  endokrynologia: "Endokrynologia",
  pulmonologia: "Pulmonologia",
  neurologia: "Neurologia",
  urologia: "Urologia",
};

export default function VisitDetailPage() {
  return (
    <AuthGuard>
      <Header />
      <VisitDetail />
    </AuthGuard>
  );
}

function VisitDetail() {
  const params = useParams<{ id: string }>();
  const visitId = params.id;
  const upsert = useVisitStore((s) => s.upsert);

  const [finalizeResult, setFinalizeResult] = useState<FinalizeResponse | null>(
    null
  );
  const [audioUploaded, setAudioUploaded] = useState(false);

  const visitQuery = useQuery({
    queryKey: ["visit", visitId],
    queryFn: () => api.getVisit(visitId),
    enabled: !!visitId,
  });

  // Mirror status into local visit store so dashboard list reflects.
  useEffect(() => {
    if (visitQuery.data) {
      upsert({
        visit_id: visitQuery.data.visit_id,
        created_at: visitQuery.data.created_at,
        specialization: visitQuery.data.specialization,
        visit_date: visitQuery.data.visit_date,
        last_known_status: visitQuery.data.status,
      });
    }
  }, [visitQuery.data, upsert]);

  if (visitQuery.isLoading) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-8">
        <div className="card animate-pulse p-6">
          <div className="h-5 w-1/3 rounded bg-slate-100" />
          <div className="mt-4 h-3 w-2/3 rounded bg-slate-100" />
        </div>
      </main>
    );
  }

  if (visitQuery.isError || !visitQuery.data) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-8">
        <div className="card p-6 text-sm text-danger">
          Nie znaleziono wizyty lub brak uprawnień.
        </div>
      </main>
    );
  }

  const visit = visitQuery.data;
  const status = visit.status;
  // The flow gates UI by visit status:
  //   draft           → show audio uploader
  //   audio_uploaded  → show finalize progress (auto-starts)
  //   transcribing/anonymizing/suggesting → show finalize progress
  //   completed       → show results (or keep finalize progress if not loaded)
  //   failed          → show finalize progress with retry
  const showUploader =
    status === "draft" && !audioUploaded;
  const showProgress =
    !finalizeResult &&
    (status === "audio_uploaded" ||
      status === "transcribing" ||
      status === "anonymizing" ||
      status === "suggesting" ||
      status === "failed" ||
      audioUploaded);
  const showResults = !!finalizeResult;

  return (
    <main className="mx-auto max-w-3xl px-6 py-8">
      <Link href="/" className="text-sm text-accent hover:underline">
        ← Wszystkie wizyty
      </Link>
      <div className="mt-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Wizyta —{" "}
            {visit.specialization
              ? SPECIALIZATION_LABELS[visit.specialization]
              : "bez specjalizacji"}
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            {visit.visit_date && (
              <>Data: {new Date(visit.visit_date).toLocaleDateString("pl-PL")} · </>
            )}
            ID: <code className="text-xs">{visit.visit_id}</code>
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-6">
        {showUploader && (
          <AudioUploader
            visitId={visit.visit_id}
            onUploaded={() => {
              setAudioUploaded(true);
              visitQuery.refetch();
            }}
          />
        )}

        {showProgress && (
          <FinalizeProgress
            visitId={visit.visit_id}
            onCompleted={(result) => setFinalizeResult(result)}
          />
        )}

        {showResults && finalizeResult && (
          <SuggestionView
            suggestion={finalizeResult.revealed_suggestion}
            unmappedTokenCount={finalizeResult.unmapped_token_count}
          />
        )}
      </div>
    </main>
  );
}
