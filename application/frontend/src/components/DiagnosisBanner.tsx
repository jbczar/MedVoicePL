"use client";

import { Stethoscope } from "lucide-react";
import type { SOAPAssessment } from "@/lib/types";

type Props = {
  assessment: SOAPAssessment;
};

export function DiagnosisBanner({ assessment }: Props) {
  const { primary_diagnosis, icd10_code, clinical_reasoning, differential } =
    assessment;

  if (!primary_diagnosis) {
    return null;
  }

  return (
    <div className="card border-l-4 border-accent">
      <div className="flex items-start gap-3">
        <div className="rounded-full bg-accent/10 p-2 text-accent">
          <Stethoscope size={20} aria-hidden />
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Rozpoznanie
          </div>
          <h2 className="mt-1 text-xl font-semibold leading-tight text-slate-900">
            {primary_diagnosis}
          </h2>
          {icd10_code && (
            <div className="mt-2 inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-xs font-mono text-slate-700">
              ICD-10: <strong>{icd10_code}</strong>
            </div>
          )}
          {clinical_reasoning && (
            <p className="mt-3 text-sm leading-relaxed text-slate-700">
              {clinical_reasoning}
            </p>
          )}
          {differential.length > 0 && (
            <div className="mt-3">
              <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Diagnostyka różnicowa
              </div>
              <ul className="mt-1 list-disc pl-5 text-sm text-slate-700">
                {differential.map((dx, i) => (
                  <li key={i}>{dx}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
