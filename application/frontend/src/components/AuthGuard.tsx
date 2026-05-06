"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { useAuth } from "@/hooks/useAuth";

/**
 * Client-side guard for authenticated routes.
 *
 * Renders nothing while the /me probe is in flight. On 401 redirects
 * to /login. On success renders children.
 *
 * We keep auth in client space (no Next middleware) because the
 * gateway runs in a different origin and the session cookie is
 * gateway-scoped — Next.js can't introspect it server-side without
 * a proxy. Single client-side check is the right tradeoff for a
 * thesis demo.
 */
export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { doctor, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && doctor === null) {
      router.replace("/login");
    }
  }, [isLoading, doctor, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-300 border-t-accent" />
      </div>
    );
  }
  if (doctor === null) return null;
  return <>{children}</>;
}
