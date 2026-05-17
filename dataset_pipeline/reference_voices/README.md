# Reference voices

Snapshot date: **2026-05-17**

Source: Mozilla Common Voice Corpus 25.0 — Polish (pl).
Downloaded from Mozilla Data Collective. License: **CC0 1.0**.
See <https://mozilladatacollective.com> and <https://commonvoice.mozilla.org/en/datasets>.

Each clip was converted to mono 22050 Hz 16-bit PCM WAV and used as an
XTTSv2 reference speaker — no PII risk since contributors opt in publicly
and Common Voice releases client_ids as anonymous hashes.

| Profile | File | Client ID | Sentence | Source |
|---|---|---|---|---|
| `doctor_male_young` | `doctor_male_young_01.wav` | `0df23af0093898dd…` | W celu zapewnienia stabilności budżetowej kładziemy również nacisk na konieczność długofalowego planowania gospodarki lokalowej | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `doctor_male_senior` | `doctor_male_senior_01.wav` | `7bfe54665af3718e…` | Pragnę pogratulować sprawnego, przemyślanego i rozważnego sposobu, w jaki wypełnia pani obowiązki przewodniczącej | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `doctor_female_young` | `doctor_female_young_01.wav` | `5a446ef7c15c5ac3…` | Oprócz działań podejmowanych przez poszczególne państwa, Europa musi wprowadzić środki mające zminimalizować negatywne konsekwencje | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `doctor_female_senior` | `doctor_female_senior_01.wav` | `c0af0a34f01c2dd7…` | Dobrodziejów pozwolił postrzelać, z którymi w pokrewieństwo miał wejść! | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_male_young` | `patient_male_young_01.wav` | `9ab58ff974131b02…` | Chciałbym państwu bardzo szczerze powiedzieć, że wspaniale naszkicował pan, panie premierze, kształt hiszpańskiej prezydencji | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_male_middle` | `patient_male_middle_01.wav` | `ef8fbeac756a6a21…` | Dlatego też chciałabym przede wszystkim podkreślić wątpliwości związane ze wszelkimi konwencjami międzynarodowymi i środowiskowymi | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_male_senior` | `patient_male_senior_01.wav` | `c1c45b55244dc231…` | To stało się ideałem mego życia i przysiągłem sobie zostać co najmniej sekretarzem sądu. | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_female_young` | `patient_female_young_01.wav` | `d9c9132152772385…` | Wniosek Komisji dopuszcza już mnożnik kosztów zanieczyszczeń, znany jako współczynnik korygujący dla obszarów górskich | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_female_middle` | `patient_female_middle_01.wav` | `9000ef434f6981f8…` | Emil obmacał wszystkie krzesła, na których wierzchu siedziała skóra, a w środku mieszkały sprężyny. | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_female_senior` | `patient_female_senior_01.wav` | `0b46ab6db3043962…` | Gdy zechcesz widzieć się ze mną, zawołaj. | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_child_male` | `patient_child_male_01.wav` | `d2ef336d55eecc7b…` | Uważamy posunięcia w kierunku centralizacji Europejskiej Służby Działań Zewnętrznych za niebezpieczne i niedemokratyczne przemiany | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |
| `patient_child_female` | `patient_child_female_01.wav` | `108d81366274a67d…` | Pilnie potrzebujemy pokojowego rozwiązania w Górskim Karabachu i na okupowanych obszarach azerbejdżańskich | [cv](https://mozilladatacollective.com/datasets/cmn27nz69015hmm0720txf781) |

## Replacing with your own recordings

Drop new files with the same names into this directory. Specs:
mono WAV, ~22050 Hz, 10-20 s, clean single speaker, speaking Polish,
no music/background. The fetch script is idempotent and skips
profiles whose WAV already exists.
