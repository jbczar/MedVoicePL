/**
 * TypeScript mirrors of the gateway/cloud Pydantic schemas.
 *
 * These are duplicated from `application/shared/src/shared/schemas/` —
 * Python-to-TS auto-codegen would need OpenAPI export + a tool. For
 * Phase 3c (single-frontend, single-backend) we hand-maintain. If the
 * Python schema changes, update here in lock-step.
 */

// Keep in sync with `application/shared/src/shared/schemas/anonymized.py::Specialization`.
export type Specialization =
  | "kardiologia"
  | "pulmonologia"
  | "medycyna_rodzinna"
  | "endokrynologia"
  | "neurologia"
  | "gastrologia"
  | "ginekologia"
  | "pediatria";

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

// --- SOAP note (revealed for doctor view; mirrors AnonymizedSOAPNote
// shape from `application/shared/src/shared/schemas/anonymized.py` but
// with `<PERSON_n>` placeholders already replaced by the gateway).

export interface SOAPSubjective {
  chief_complaint: string;
  history_of_present_illness: string;
  review_of_systems: string[];
  medical_history: string[];
  medications_current: string[];
  allergies: string[];
  provenance: number[];
}

export interface SOAPVital {
  name: string;
  value: string;
  unit: string | null;
}

export interface SOAPLabMention {
  name: string;
  value: string | null;
  unit: string | null;
}

export interface SOAPObjective {
  physical_exam: string[];
  vitals: SOAPVital[];
  labs_mentioned: SOAPLabMention[];
  provenance: number[];
}

export interface SOAPAssessment {
  primary_diagnosis: string;
  icd10_code: string | null;
  differential: string[];
  clinical_reasoning: string;
  provenance: number[];
}

export interface SOAPPlan {
  medications_prescribed: string[];
  diagnostic_tests: string[];
  referrals: string[];
  follow_up: string | null;
  patient_education: string[];
  provenance: number[];
}

export interface SOAPMetadata {
  anonymizer_version: string;
  n_entities_per_type?: Record<string, number>;
  visit_date?: string;
  specialization?: Specialization;
  session_id?: string;
}

export interface SOAPNote {
  subjective: SOAPSubjective;
  objective: SOAPObjective;
  assessment: SOAPAssessment;
  plan: SOAPPlan;
  red_flags: string[];
  red_flags_provenance?: number[];
  metadata: SOAPMetadata;
}

// --- Transcript (un-anonymized; doctor sees verbatim STT output). ---

export interface TranscriptWordTiming {
  word: string;
  start_s: number;
  end_s: number;
}

export interface TranscriptSegment {
  speaker: "doctor" | "patient" | string;
  start_s: number;
  end_s: number;
  text: string;
  words?: TranscriptWordTiming[];
}

export interface TranscriptMetadata {
  engine?: string;
  model?: string;
  language?: string;
}

export interface Transcript {
  scenario_id: string;
  audio_path: string;
  audio_sha256: string;
  duration_s: number;
  segments: TranscriptSegment[];
  metadata: TranscriptMetadata;
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
