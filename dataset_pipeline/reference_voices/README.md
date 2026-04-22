# Reference voices

Snapshot date: **2026-04-22**

Source: Mozilla Common Voice Corpus 25.0 — Polish (pl).
Downloaded from Mozilla Data Collective. License: **CC0 1.0**.
See <https://mozilladatacollective.com> and <https://commonvoice.mozilla.org/en/datasets>.

Each clip was converted to mono 22050 Hz 16-bit PCM WAV and used as an
XTTSv2 reference speaker — no PII risk since contributors opt in publicly
and Common Voice releases client_ids as anonymous hashes.

| Profile | File | Client ID | Sentence | Source |
|---|---|---|---|---|
| `doctor_male` | `doctor_male_01.wav` | `ef8fbeac756a6a21…` | Dlatego też chciałabym przede wszystkim podkreślić wątpliwości związane ze wszelkimi konwencjami międzynarodowymi i środowiskowymi | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `doctor_female` | `doctor_female_01.wav` | `c666546baeecd06b…` | Wiele miejsca zajmują materiały poświęconej rozwojowi oraz działalności gromady żydowskiej miasta Czerniowców. | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_male` | `patient_male_01.wav` | `0df23af0093898dd…` | W celu zapewnienia stabilności budżetowej kładziemy również nacisk na konieczność długofalowego planowania gospodarki lokalowej | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_female` | `patient_female_01.wav` | `a6c35f97ef341789…` | Po drugie, jakość wyprodukowanej bawełny poprawi się, a branża odziarniania bawełny zostanie unowocześniona | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |

## Replacing with your own recordings

Drop new files with the same names into this directory. Specs:
mono WAV, ~22050 Hz, 10-20 s, clean single speaker, speaking Polish,
no music/background. The fetch script is idempotent and skips
profiles whose WAV already exists.
