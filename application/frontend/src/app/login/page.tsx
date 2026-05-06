"use client";

import { Stethoscope } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth, useLoginMutation } from "@/hooks/useAuth";
import { ApiError } from "@/lib/api";

export default function LoginPage() {
  const { doctor, isLoading } = useAuth();
  const router = useRouter();
  const login = useLoginMutation();
  const [username, setUsername] = useState("demo_doctor");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // If already authenticated, bounce to dashboard.
  useEffect(() => {
    if (!isLoading && doctor) router.replace("/");
  }, [isLoading, doctor, router]);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);
    try {
      await login.mutateAsync({ username, password });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setErrorMessage("Nieprawidłowa nazwa użytkownika lub hasło.");
      } else {
        setErrorMessage(
          "Nie udało się połączyć z serwerem. Sprawdź czy gateway jest uruchomiony."
        );
      }
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="card w-full max-w-md p-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-lg bg-accent-subtle p-2 text-accent">
            <Stethoscope className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">MedVoicePL</h1>
            <p className="text-sm text-slate-500">Pulpit lekarza</p>
          </div>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="label" htmlFor="username">
              Nazwa użytkownika
            </label>
            <input
              id="username"
              className="input"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="label" htmlFor="password">
              Hasło
            </label>
            <input
              id="password"
              type="password"
              className="input"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
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

          <button
            type="submit"
            className="btn-primary w-full"
            disabled={login.isPending}
          >
            {login.isPending ? "Logowanie…" : "Zaloguj"}
          </button>
        </form>

        <p className="mt-6 text-xs text-slate-500">
          Demo: <code className="rounded bg-slate-100 px-1 py-0.5">demo_doctor</code>{" "}
          / <code className="rounded bg-slate-100 px-1 py-0.5">demo_password</code>
        </p>
      </div>
    </main>
  );
}
