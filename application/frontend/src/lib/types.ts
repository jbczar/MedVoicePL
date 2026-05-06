/**
 * TypeScript mirrors of the gateway/cloud Pydantic schemas.
 *
 * These are duplicated from `application/shared/src/shared/schemas/` —
 * Python-to-TS auto-codegen would need OpenAPI export + a tool. For
 * Phase 3c (single-frontend, single-backend) we hand-maintain. If the
 * Python schema changes, update here in lock-step.
 */

export type Specialization =
  | "kardiologia"
  | "endokrynologia"
  | "pulmonologia"
  | "neurologia"
  | "urologia";

export type VisitStatus =
  | "draft"
  | "audio_uploaded"
  | "transcribing"
  | "structuring"
  | "anonymizing"
  | "suggesting"
  | "completed"
  | "failed";

export interface VisitSummary {
  visit_id: string;
  status: VisitStatus;
  created_at: string;
  completed_at: string | null;
  specialization: Specialization | null;
  visit_date: string | null;
  duration_ms: number | null;
  error: string | null;
}

export interface DoctorIdentity {
  username: string;
  full_name: string;
}

export interface CreateVisitResponse {
  visit_id: string;
  status: VisitStatus;
}

export interface AudioUploadResponse {
  visit_id: string;
  status: VisitStatus;
  audio_sha256: string;
}

// --- Cloud agent types ---

export type ReimbursementLevel =
  | "100%"
  | "70%"
  | "50%"
  | "ryczałt"
  | "nierefundowane";

export interface DrugSuggestion {
  name: string;
  atc_code: string;
  reimbursement: ReimbursementLevel;
  patient_copay_pln: number | string | null; // Decimal serialized to string in JSON
  indication_restrictions: string[];
  provenance: string[];
  rationale: string;
}

export interface Referral {
  specialty: string;
  rationale: string;
}

export interface DiagnosticTest {
  name: string;
  rationale: string;
  urgency: string | null;
}

export interface AgentMetadata {
  agent_version: string;
  llm_model: string;
  nfz_snapshot_date: string;
  chpl_corpus_size: number;
  duration_s: number;
}

export interface AgentSuggestion {
  drugs: DrugSuggestion[];
  referrals: Referral[];
  tests: DiagnosticTest[];
  notes: string[];
  metadata: AgentMetadata;
}

// --- Anonymized SOAP (subset used by the doctor's UI) ---

export interface AnonymizedSubjective {
  chief_complaint: string;
  history_of_present_illness: string;
  review_of_systems: string[];
  past_medical_history: string[];
  medications: string[];
  allergies: string[];
  social_history: string;
  family_history: string;
  provenance: number[];
}

export interface AnonymizedVital {
  name: string;
  value: string;
  unit: string | null;
}

export interface AnonymizedLabMention {
  name: string;
  value: string | null;
  unit: string | null;
  context: string;
}

export interface AnonymizedObjective {
  vitals: AnonymizedVital[];
  physical_examination: string;
  labs_mentioned: AnonymizedLabMention[];
  imaging_mentioned: string[];
  provenance: number[];
}

export interface AnonymizedAssessment {
  primary_diagnosis: string;
  differential_diagnoses: string[];
  icd10_code: string | null;
  clinical_reasoning: string;
  provenance: number[];
}

export interface AnonymizedPlan {
  medications_prescribed: string[];
  diagnostic_tests: string[];
  referrals: string[];
  follow_up: string;
  patient_education: string;
  provenance: number[];
}

export interface AnonymizedMetadata {
  anonymizer_version: string;
  n_entities_per_type: Record<string, number>;
  visit_date: string;
  specialization: Specialization;
  session_id: string;
}

export interface AnonymizedSOAPNote {
  subjective: AnonymizedSubjective;
  objective: AnonymizedObjective;
  assessment: AnonymizedAssessment;
  plan: AnonymizedPlan;
  red_flags: string[];
  metadata: AnonymizedMetadata;
}

export interface FinalizeResponse {
  visit: VisitSummary;
  suggestion: AgentSuggestion;
  revealed_suggestion: AgentSuggestion;
  unmapped_token_count: number;
}

// --- Health probe ---

export interface HealthResponse {
  status: "ok" | "degraded";
  on_premise_reachable: boolean;
  cloud_reachable: boolean;
  version: string;
}
