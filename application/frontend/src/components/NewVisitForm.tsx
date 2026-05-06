"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";

import * as api from "@/lib/api";
import type { Specialization } from "@/lib/types";
import { useVisitStore } from "@/lib/visitStore";

const SPECIALIZATIONS: { value: Specialization; label: string }[] = [
  { value: "kardiologia", label: "Kardiologia" },
  { value: "endokrynologia", label: "Endokrynologia" },
  { value: "pulmonologia", label: "Pulmonologia" },
  { value: "neurologia", label: "Neurologia" },
  { value: "urologia", label: "Urologia" },
];

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

export function NewVisitForm() {
  const router = useRouter();
  const upsert = useVisitStore((s) => s.upsert);
  const [specialization, setSpecialization] = useState<Specialization>("endokrynologia");
  const [visitDate, setVisitDate] = useState<string>(todayISO());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const createMutation = useMutation({
    mutationFn: () =>
      api.createVisit({
        specialization,
        visit_date: visitDate,
      }),
    onSuccess: (resp) => {
      upsert({
        visit_id: resp.visit_id,
        created_at: new Date().toISOString(),
        specialization,
        visit_date: visitDate,
        last_known_status: resp.status,
      });
      router.push(`/visits/${resp.visit_id}`);
    },
    onError: () => {
      setErrorMessage("Nie udało się utworzyć wizyty. Sprawdź połączenie z gatewayem.");
    },
  });

  function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setErrorMessage(null);
    createMutation.mutate();
  }

  return (
    <form className="space-y-5" onSubmit={onSubmit}>
      <div>
        <label className="label" htmlFor="specialization">
          Specjalizacja
        </label>
        <select
          id="specialization"
          className="input"
          value={specialization}
          onChange={(e) => setSpecialization(e.target.value as Specialization)}
        >
          {SPECIALIZATIONS.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="label" htmlFor="visit_date">
          Data wizyty
        </label>
        <input
          id="visit_date"
          type="date"
          className="input"
          value={visitDate}
          onChange={(e) => setVisitDate(e.target.value)}
          max={todayISO()}
          required
        />
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="rounded-md border border-danger/30 bg-red-50 px-3 py-2 text-sm text-danger"
        >
          {errorMessage}
        </div>
      )}

      <div className="flex items-center justify-end gap-2">
        <button
          type="button"
          className="btn-secondary"
          onClick={() => router.push("/")}
          disabled={createMutation.isPending}
        >
          Anuluj
        </button>
        <button type="submit" className="btn-primary" disabled={createMutation.isPending}>
          {createMutation.isPending ? "Tworzenie…" : "Utwórz wizytę"}
        </button>
      </div>
    </form>
  );
}
