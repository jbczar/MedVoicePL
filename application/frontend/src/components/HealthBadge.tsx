"use client";

import { useQuery } from "@tanstack/react-query";

import * as api from "@/lib/api";
import { cn } from "@/lib/cn";

/** Tiny chip showing whether on-prem and cloud are reachable. */
export function HealthBadge() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    staleTime: 10_000,
  });

  const status = isError || !data ? "unreachable" : data.status;

  const label =
    status === "ok"
      ? "Wszystkie usługi online"
      : status === "degraded"
        ? data?.on_premise_reachable
          ? "Cloud niedostępny"
          : "On-prem niedostępny"
        : "Gateway niedostępny";

  const dotClass =
    status === "ok"
      ? "bg-ok"
      : status === "degraded"
        ? "bg-warn"
        : "bg-danger";

  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600">
      <span
        className={cn("inline-block h-2 w-2 rounded-full", dotClass)}
        aria-hidden
      />
      {label}
    </div>
  );
}
