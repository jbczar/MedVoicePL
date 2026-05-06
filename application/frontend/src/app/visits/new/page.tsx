"use client";

import Link from "next/link";

import { AuthGuard } from "@/components/AuthGuard";
import { Header } from "@/components/Header";
import { NewVisitForm } from "@/components/NewVisitForm";

export default function NewVisitPage() {
  return (
    <AuthGuard>
      <Header />
      <main className="mx-auto max-w-xl px-6 py-8">
        <Link href="/" className="text-sm text-accent hover:underline">
          ← Wstecz
        </Link>
        <h1 className="mt-3 text-2xl font-semibold tracking-tight">Nowa wizyta</h1>
        <p className="mt-1 text-sm text-slate-500">
          Po utworzeniu wizyty przejdziesz do nagrywania audio.
        </p>
        <div className="card mt-6 p-6">
          <NewVisitForm />
        </div>
      </main>
    </AuthGuard>
  );
}
