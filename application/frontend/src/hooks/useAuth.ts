"use client";

import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";

import * as api from "@/lib/api";
import type { DoctorIdentity } from "@/lib/types";

const ME_KEY = ["me"] as const;

/**
 * Returns the current doctor identity (or null when unauthenticated).
 *
 * `isLoading` flips while we resolve the cookie session against the
 * gateway. The /me call is the source of truth — there's no client-side
 * token, the cookie is httpOnly and we trust the server.
 */
export function useAuth() {
  const query = useQuery({
    queryKey: ME_KEY,
    queryFn: async (): Promise<DoctorIdentity | null> => {
      try {
        return await api.me();
      } catch (err: unknown) {
        // 401 = not authenticated; map to null instead of throwing.
        if (err && typeof err === "object" && "status" in err && err.status === 401) {
          return null;
        }
        throw err;
      }
    },
    retry: false,
    staleTime: 60_000,
  });
  return {
    doctor: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
  };
}

/** `await loginMutation.mutateAsync({username, password})` then redirect. */
export function useLoginMutation() {
  const qc = useQueryClient();
  const router = useRouter();
  return useMutation({
    mutationFn: ({ username, password }: { username: string; password: string }) =>
      api.login(username, password),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ME_KEY });
      router.push("/");
    },
  });
}

export function useLogoutMutation() {
  const qc = useQueryClient();
  const router = useRouter();
  return useMutation({
    mutationFn: () => api.logout(),
    onSuccess: () => {
      qc.setQueryData(ME_KEY, null);
      qc.clear();
      router.push("/login");
    },
  });
}
