"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import type { Specialization, VisitStatus } from "./types";

/**
 * Local-only history of visits the doctor has interacted with in this
 * browser. The gateway doesn't expose a list endpoint (per Phase 3b
 * design), so we maintain a thin client-side index. Persisted to
 * localStorage so the dashboard isn't empty on refresh.
 *
 * Each entry is a thin pointer — the authoritative status comes from
 * `GET /v1/visits/{id}` on demand.
 */
export interface VisitIndexEntry {
  visit_id: string;
  created_at: string; // ISO
  specialization: Specialization | null;
  visit_date: string | null;
  last_known_status: VisitStatus;
}

interface VisitStore {
  visits: VisitIndexEntry[];
  upsert: (entry: VisitIndexEntry) => void;
  remove: (visit_id: string) => void;
  clear: () => void;
}

export const useVisitStore = create<VisitStore>()(
  persist(
    (set) => ({
      visits: [],
      upsert: (entry) =>
        set((state) => {
          const without = state.visits.filter(
            (v) => v.visit_id !== entry.visit_id
          );
          return { visits: [entry, ...without] };
        }),
      remove: (visit_id) =>
        set((state) => ({
          visits: state.visits.filter((v) => v.visit_id !== visit_id),
        })),
      clear: () => set({ visits: [] }),
    }),
    {
      name: "medvoicepl.visits",
      storage: createJSONStorage(() => localStorage),
    }
  )
);
