"use client";

import { Plus, Stethoscope } from "lucide-react";
import Link from "next/link";

import { AuthGuard } from "@/components/AuthGuard";
import { Header } from "@/components/Header";
import { HealthBadge } from "@/components/HealthBadge";
import { cn } from "@/lib/cn";
import { useVisitStore, type VisitIndexEntry } from "@/lib/visitStore";

const SPECIALIZATION_LABELS: Record<string, string> = {
  kardiologia: "Kardiologia",
  endokrynologia: "Endokrynologia",
  pulmonologia: "Pulmonologia",
  neurologia: "Neurologia",
  urologia: "Urologia",
};

const STATUS_LABELS: Record<string, string> = {
  draft: "Szkic",
  audio_uploaded: "Audio przesłane",
  transcribing: "Transkrypcja",
  structuring: "Strukturyzacja",
  anonymizing: "Anonimizacja",
  suggesting: "Generowanie sugestii",
  completed: "Ukończona",
  failed: "Błąd",
};

const STATUS_COLOR: Record<string, string> = {
  draft: "bg-slate-100 text-slate-700",
  audio_uploaded: "bg-sky-100 text-sky-700",
  transcribing: "bg-sky-100 text-sky-700",
  structuring: "bg-sky-100 text-sky-700",
  anonymizing: "bg-sky-100 text-sky-700",
  suggesting: "bg-sky-100 text-sky-700",
  completed: "bg-emerald-100 text-emerald-700",
  failed: "bg-red-100 text-red-700",
};

function formatVisitDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("pl-PL");
}

function formatRelative(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "przed chwilą";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min temu`;
  if (seconds < 86_400) return `${Math.floor(seconds / 3600)} godz. temu`;
  return new Date(iso).toLocaleDateString("pl-PL");
}

function VisitRow({ visit }: { visit: VisitIndexEntry }) {
  return (
    <Link
      href={`/visits/${visit.visit_id}`}
      className="flex items-center justify-between border-b border-slate-100 px-6 py-4 last:border-b-0 hover:bg-slate-50"
    >
      <div className="flex items-start gap-3">
        <div className="rounded-md bg-slate-100 p-1.5 text-slate-500">
          <Stethoscope className="h-4 w-4" />
        </div>
        <div>
          <div className="text-sm font-medium text-slate-900">
            {visit.specialization
              ? SPECIALIZATION_LABELS[visit.specialization]
              : "Bez specjalizacji"}
          </div>
          <div className="mt-0.5 text-xs text-slate-500">
            {formatVisitDate(visit.visit_date)} · utworzono {formatRelative(visit.created_at)}
          </div>
        </div>
      </div>
      <span
        className={cn(
          "rounded-full px-2.5 py-1 text-xs font-medium",
          STATUS_COLOR[visit.last_known_status] ?? "bg-slate-100 text-slate-700"
        )}
      >
        {STATUS_LABELS[visit.last_known_status] ?? visit.last_known_status}
      </span>
    </Link>
  );
}

export default function DashboardPage() {
  return (
    <AuthGuard>
      <Header />
      <Dashboard />
    </AuthGuard>
  );
}

function Dashboard() {
  const visits = useVisitStore((s) => s.visits);

  return (
    <main className="mx-auto max-w-5xl px-6 py-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Wizyty</h1>
          <p className="mt-1 text-sm text-slate-500">
            Lokalna historia tej przeglądarki. Każda wizyta dziedziczy aktualny status z gatewaya.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <HealthBadge />
          <Link href="/visits/new" className="btn-primary">
            <Plus className="h-4 w-4" />
            Nowa wizyta
          </Link>
        </div>
      </div>

      <section className="card overflow-hidden">
        {visits.length === 0 ? (
          <div className="px-6 py-16 text-center">
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-500">
              <Stethoscope className="h-5 w-5" />
            </div>
            <p className="text-sm text-slate-600">
              Brak wizyt. Kliknij <span className="font-medium">Nowa wizyta</span>, aby zacząć.
            </p>
          </div>
        ) : (
          <div>
            {visits.map((v) => (
              <VisitRow key={v.visit_id} visit={v} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
