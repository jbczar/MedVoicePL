"use client";

import { LogOut, Stethoscope } from "lucide-react";
import Link from "next/link";

import { useAuth, useLogoutMutation } from "@/hooks/useAuth";

export function Header() {
  const { doctor } = useAuth();
  const logout = useLogoutMutation();

  return (
    <header className="border-b border-slate-200 bg-card-bg">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
        <Link href="/" className="flex items-center gap-2">
          <div className="rounded-md bg-accent-subtle p-1.5 text-accent">
            <Stethoscope className="h-4 w-4" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-tight">MedVoicePL</div>
            <div className="text-xs text-slate-500">Pulpit lekarza</div>
          </div>
        </Link>

        <div className="flex items-center gap-3">
          {doctor && (
            <span className="hidden text-sm text-slate-600 sm:inline">
              {doctor.full_name}
            </span>
          )}
          <button
            type="button"
            className="btn-secondary text-sm"
            onClick={() => logout.mutate()}
            disabled={logout.isPending}
            aria-label="Wyloguj"
          >
            <LogOut className="h-4 w-4" />
            <span>Wyloguj</span>
          </button>
        </div>
      </div>
    </header>
  );
}
