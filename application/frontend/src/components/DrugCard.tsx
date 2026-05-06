"use client";

import { CheckCircle2, ChevronDown, Pill } from "lucide-react";
import { useState } from "react";

import type { DrugSuggestion, ReimbursementLevel } from "@/lib/types";
import { cn } from "@/lib/cn";

const REFUND_STYLES: Record<
  ReimbursementLevel,
  { label: string; className: string; description: string }
> = {
  "100%": {
    label: "Refundacja 100%",
    className: "bg-emerald-50 text-emerald-700 border-emerald-200",
    description: "Lek w pełni refundowany przez NFZ.",
  },
  "70%": {
    label: "Refundacja 70%",
    className: "bg-sky-50 text-sky-700 border-sky-200",
    description: "Pacjent dopłaca 30% ceny.",
  },
  "50%": {
    label: "Refundacja 50%",
    className: "bg-sky-50 text-sky-700 border-sky-200",
    description: "Pacjent dopłaca 50% ceny.",
  },
  "ryczałt": {
    label: "Ryczałt",
    className: "bg-amber-50 text-amber-800 border-amber-200",
    description: "Stała opłata za opakowanie (zwykle 3,20 zł).",
  },
  "nierefundowane": {
    label: "Brak refundacji",
    className: "bg-slate-50 text-slate-600 border-slate-200",
    description: "Lek nie znajduje się w wykazie NFZ — pacjent płaci 100%.",
  },
};

interface Props {
  drug: DrugSuggestion;
  onProvenanceClick?: (paths: string[]) => void;
  highlighted?: boolean;
}

export function DrugCard({ drug, onProvenanceClick, highlighted }: Props) {
  const [showDetails, setShowDetails] = useState(false);
  const refund = REFUND_STYLES[drug.reimbursement];
  const copay = drug.patient_copay_pln;

  return (
    <article
      className={cn(
        "card overflow-hidden transition-shadow",
        highlighted && "ring-2 ring-accent ring-offset-2 ring-offset-page-bg"
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3 p-5">
        <div className="flex items-start gap-3">
          <div className="rounded-md bg-accent-subtle p-2 text-accent">
            <Pill className="h-5 w-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-baseline gap-x-2">
              <h3 className="text-lg font-semibold capitalize">{drug.name}</h3>
              <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono text-slate-600">
                {drug.atc_code}
              </code>
            </div>
            <p className="mt-1 text-sm text-slate-600">{drug.rationale}</p>
          </div>
        </div>
        <span
          className={cn(
            "shrink-0 rounded-full border px-3 py-1 text-xs font-medium",
            refund.className
          )}
          title={refund.description}
        >
          {refund.label}
        </span>
      </div>

      {(copay !== null && copay !== undefined) && (
        <div className="border-t border-slate-100 bg-slate-50/60 px-5 py-2 text-xs text-slate-600">
          Szacunkowa dopłata pacjenta:{" "}
          <span className="font-medium text-slate-800">
            {typeof copay === "string" ? copay : copay.toFixed(2)} zł
          </span>
        </div>
      )}

      {drug.indication_restrictions.length > 0 && (
        <div className="border-t border-slate-100 px-5 py-3 text-xs">
          <div className="font-medium text-slate-700">Wskazania objęte refundacją:</div>
          <ul className="mt-1 list-inside list-disc text-slate-600">
            {drug.indication_restrictions.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      <button
        type="button"
        className="flex w-full items-center justify-between border-t border-slate-100 px-5 py-3 text-xs font-medium text-slate-500 hover:bg-slate-50"
        onClick={() => {
          setShowDetails((v) => !v);
          if (!showDetails) onProvenanceClick?.(drug.provenance);
        }}
        aria-expanded={showDetails}
      >
        <span>
          Podstawa decyzji ({drug.provenance.length})
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            showDetails && "rotate-180"
          )}
        />
      </button>

      {showDetails && (
        <ul className="border-t border-slate-100 bg-slate-50/60 px-5 py-3 text-xs text-slate-600">
          {drug.provenance.map((path, i) => (
            <li key={i} className="flex items-center gap-2 py-0.5">
              <CheckCircle2 className="h-3 w-3 text-emerald-500" />
              <code className="text-slate-700">{path}</code>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
