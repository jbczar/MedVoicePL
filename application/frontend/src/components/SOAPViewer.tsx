"use client";

import { useState } from "react";
import { ChevronDown, FileText } from "lucide-react";
import { cn } from "@/lib/cn";
import type {
  SOAPAssessment,
  SOAPNote,
  SOAPObjective,
  SOAPPlan,
  SOAPSubjective,
} from "@/lib/types";

type Props = {
  soap: SOAPNote;
};

export function SOAPViewer({ soap }: Props) {
  return (
    <section className="card">
      <header className="flex items-center gap-2 border-b border-slate-200 pb-3">
        <FileText size={18} className="text-accent" aria-hidden />
        <h2 className="text-base font-semibold text-slate-900">Notatka SOAP</h2>
        <span className="text-xs text-slate-500">— rozmowa po strukturyzacji</span>
      </header>
      <div className="mt-2 divide-y divide-slate-100">
        <Section
          letter="S"
          title="Subjective — wywiad"
          subtitle={soap.subjective.chief_complaint}
        >
          <SubjectiveBody data={soap.subjective} />
        </Section>
        <Section
          letter="O"
          title="Objective — badanie"
          subtitle={objectiveSubtitle(soap.objective)}
        >
          <ObjectiveBody data={soap.objective} />
        </Section>
        <Section
          letter="A"
          title="Assessment — ocena"
          subtitle={soap.assessment.primary_diagnosis}
        >
          <AssessmentBody data={soap.assessment} />
        </Section>
        <Section
          letter="P"
          title="Plan — postępowanie"
          subtitle={planSubtitle(soap.plan)}
        >
          <PlanBody data={soap.plan} />
        </Section>
      </div>
    </section>
  );
}

type SectionProps = {
  letter: "S" | "O" | "A" | "P";
  title: string;
  subtitle?: string;
  children: React.ReactNode;
};

function Section({ letter, title, subtitle, children }: SectionProps) {
  const [open, setOpen] = useState(letter === "A"); // Assessment expanded by default

  return (
    <div className="py-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-3 rounded-md px-1 text-left transition hover:bg-slate-50"
        aria-expanded={open}
      >
        <div className="flex items-start gap-3">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent/10 text-sm font-bold text-accent">
            {letter}
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-900">{title}</div>
            {subtitle && (
              <div className="mt-0.5 line-clamp-1 text-xs text-slate-500">
                {subtitle}
              </div>
            )}
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
      {open && <div className="ml-10 mt-3 space-y-3 text-sm text-slate-700">{children}</div>}
    </div>
  );
}

// --- Section bodies ---

function SubjectiveBody({ data }: { data: SOAPSubjective }) {
  return (
    <>
      <Field label="Główna dolegliwość">{data.chief_complaint}</Field>
      <Field label="Historia obecnej choroby">
        {data.history_of_present_illness}
      </Field>
      <ListField label="Wywiad układów" items={data.review_of_systems} />
      <ListField label="Choroby przewlekłe" items={data.medical_history} />
      <ListField
        label="Leki przyjmowane"
        items={data.medications_current}
      />
      <ListField label="Alergie" items={data.allergies} />
    </>
  );
}

function ObjectiveBody({ data }: { data: SOAPObjective }) {
  return (
    <>
      <ListField label="Badanie przedmiotowe" items={data.physical_exam} />
      {data.vitals.length > 0 && (
        <div>
          <FieldLabel>Parametry życiowe</FieldLabel>
          <ul className="mt-1 space-y-0.5 text-sm">
            {data.vitals.map((v, i) => (
              <li key={i}>
                <span className="font-medium">{v.name.toUpperCase()}:</span>{" "}
                {v.value}
                {v.unit ? ` ${v.unit}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.labs_mentioned.length > 0 && (
        <div>
          <FieldLabel>Badania laboratoryjne (wzmiankowane)</FieldLabel>
          <ul className="mt-1 list-disc pl-5 text-sm">
            {data.labs_mentioned.map((lab, i) => (
              <li key={i}>
                {lab.name}
                {lab.value && (
                  <>
                    : <span className="font-mono">{lab.value}</span>
                    {lab.unit ? ` ${lab.unit}` : ""}
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

function AssessmentBody({ data }: { data: SOAPAssessment }) {
  return (
    <>
      <Field label="Rozpoznanie wstępne">{data.primary_diagnosis}</Field>
      {data.icd10_code && (
        <Field label="ICD-10">
          <span className="font-mono">{data.icd10_code}</span>
        </Field>
      )}
      {data.clinical_reasoning && (
        <Field label="Uzasadnienie kliniczne">{data.clinical_reasoning}</Field>
      )}
      <ListField
        label="Diagnostyka różnicowa"
        items={data.differential}
      />
    </>
  );
}

function PlanBody({ data }: { data: SOAPPlan }) {
  return (
    <>
      <ListField
        label="Leki przepisane (wg lekarza, w trakcie wizyty)"
        items={data.medications_prescribed}
      />
      <ListField label="Badania zlecone" items={data.diagnostic_tests} />
      <ListField label="Skierowania" items={data.referrals} />
      {data.follow_up && <Field label="Kontrola">{data.follow_up}</Field>}
      <ListField label="Zalecenia dla pacjenta" items={data.patient_education} />
    </>
  );
}

// --- Field primitives ---

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
      {children}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  if (!children) {
    return null;
  }
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <div className="mt-0.5 text-sm text-slate-700">{children}</div>
    </div>
  );
}

function ListField({ label, items }: { label: string; items: string[] }) {
  if (!items || items.length === 0) {
    return null;
  }
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <ul className="mt-1 list-disc pl-5 text-sm">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

// --- Helpers ---

function objectiveSubtitle(o: SOAPObjective): string {
  const bits: string[] = [];
  if (o.physical_exam.length > 0) bits.push(`${o.physical_exam.length} obserwacji`);
  if (o.vitals.length > 0) bits.push(`${o.vitals.length} vitals`);
  if (o.labs_mentioned.length > 0) bits.push(`${o.labs_mentioned.length} badań`);
  return bits.length > 0 ? bits.join(" · ") : "brak danych";
}

function planSubtitle(p: SOAPPlan): string {
  const bits: string[] = [];
  if (p.medications_prescribed.length > 0)
    bits.push(`${p.medications_prescribed.length} leków`);
  if (p.diagnostic_tests.length > 0)
    bits.push(`${p.diagnostic_tests.length} badań`);
  if (p.referrals.length > 0) bits.push(`${p.referrals.length} skierowań`);
  if (p.follow_up) bits.push("kontrola");
  return bits.length > 0 ? bits.join(" · ") : "brak danych";
}
