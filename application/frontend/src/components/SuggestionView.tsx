"use client";

import {
  AlertTriangle,
  ArrowRightCircle,
  ClipboardCheck,
  ClipboardList,
  Info,
  Pill,
} from "lucide-react";
import { useState } from "react";

import { DrugCard } from "@/components/DrugCard";
import type { AgentSuggestion } from "@/lib/types";

interface Props {
  suggestion: AgentSuggestion;
  unmappedTokenCount: number;
}

export function SuggestionView({ suggestion, unmappedTokenCount }: Props) {
  const [highlightPaths, setHighlightPaths] = useState<string[]>([]);

  return (
    <div className="space-y-6">
      <SectionHeader
        icon={<Pill className="h-4 w-4" />}
        title="Sugerowane leki"
        count={suggestion.drugs.length}
      />
      {suggestion.drugs.length > 0 ? (
        <div className="space-y-3">
          {suggestion.drugs.map((drug) => (
            <DrugCard
              key={`${drug.atc_code}-${drug.name}`}
              drug={drug}
              onProvenanceClick={setHighlightPaths}
            />
          ))}
        </div>
      ) : (
        <EmptyHint message="Brak sugestii leków — diagnoza wymaga potwierdzenia lub farmakoterapia nie jest wskazana." />
      )}

      {suggestion.notes.length > 0 && (
        <NoteList notes={suggestion.notes} />
      )}

      {unmappedTokenCount > 0 && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="h-4 w-4" />
            Nieodszyfrowane oznaczenia ({unmappedTokenCount})
          </div>
          <p className="mt-1 text-xs">
            Część placeholderów PII nie została odszyfrowana. Skontaktuj się z administratorem.
          </p>
        </div>
      )}

      {suggestion.tests.length > 0 && (
        <div>
          <SectionHeader
            icon={<ClipboardList className="h-4 w-4" />}
            title="Zalecane badania"
            count={suggestion.tests.length}
          />
          <ul className="mt-3 space-y-2">
            {suggestion.tests.map((test, i) => (
              <li key={i} className="card flex items-start justify-between gap-3 px-4 py-3">
                <div>
                  <div className="text-sm font-medium text-slate-800">{test.name}</div>
                  {test.rationale && (
                    <div className="mt-0.5 text-xs text-slate-500">{test.rationale}</div>
                  )}
                </div>
                {test.urgency && (
                  <span className="shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700">
                    {test.urgency}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {suggestion.referrals.length > 0 && (
        <div>
          <SectionHeader
            icon={<ArrowRightCircle className="h-4 w-4" />}
            title="Skierowania"
            count={suggestion.referrals.length}
          />
          <ul className="mt-3 space-y-2">
            {suggestion.referrals.map((ref, i) => (
              <li key={i} className="card px-4 py-3">
                <div className="text-sm font-medium capitalize text-slate-800">
                  {ref.specialty}
                </div>
                {ref.rationale && (
                  <div className="mt-0.5 text-xs text-slate-500">{ref.rationale}</div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <FooterMetadata suggestion={suggestion} highlightPaths={highlightPaths} />
    </div>
  );
}

function SectionHeader({
  icon,
  title,
  count,
}: {
  icon: React.ReactNode;
  title: string;
  count: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="rounded-md bg-slate-100 p-1.5 text-slate-500">{icon}</span>
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>
      <span className="text-xs text-slate-500">({count})</span>
    </div>
  );
}

function NoteList({ notes }: { notes: string[] }) {
  return (
    <div className="rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
      <div className="flex items-center gap-2 font-medium">
        <Info className="h-4 w-4" />
        Uwagi agenta
      </div>
      <ul className="mt-1 list-inside list-disc text-xs">
        {notes.map((n, i) => (
          <li key={i}>{n}</li>
        ))}
      </ul>
    </div>
  );
}

function EmptyHint({ message }: { message: string }) {
  return (
    <div className="card flex items-start gap-3 px-4 py-3">
      <ClipboardCheck className="h-5 w-5 shrink-0 text-slate-400" />
      <p className="text-sm text-slate-600">{message}</p>
    </div>
  );
}

function FooterMetadata({
  suggestion,
  highlightPaths,
}: {
  suggestion: AgentSuggestion;
  highlightPaths: string[];
}) {
  const m = suggestion.metadata;
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500">
      <div>
        Wygenerowane przez {m.llm_model} (agent {m.agent_version}) ·{" "}
        wykaz NFZ {m.nfz_snapshot_date} · {m.duration_s.toFixed(1)} s
      </div>
      {highlightPaths.length > 0 && (
        <div className="mt-1">
          Aktywne podświetlenie: {highlightPaths.join(", ")}
        </div>
      )}
    </div>
  );
}
