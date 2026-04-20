"""System and few-shot prompt templates for scenario generation.

Few-shot examples anchor the clinical register and structure. They are
hand-written — do not replace them with model-generated output, or drift
will compound across generations.
"""

from __future__ import annotations

from .schema import DialogueTurn, ScenarioModel, SpeakerRole, Specialization

SYSTEM_PROMPT_PL = """\
Jesteś ekspertem generującym realistyczne, syntetyczne transkrypcje wizyt \
lekarskich w języku polskim. Każda wizyta to dialog lekarza z pacjentem \
w gabinecie. Twoim zadaniem jest wygenerowanie pojedynczego scenariusza w \
formacie JSON zgodnym z podanym schematem.

Zasady jakości:
1. Dialog musi być naturalny w języku polskim mówionym. Pacjent używa \
języka potocznego, lekarz — profesjonalnej terminologii medycznej.
2. Pierwsza kwestia zawsze należy do lekarza (powitanie, pytanie otwierające).
3. Kwestie muszą występować naprzemiennie: lekarz → pacjent → lekarz → ...
4. Liczba tur: od 6 do 20. Dobieraj pod złożoność przypadku.
5. Wywiad medyczny musi być spójny: objawy → badanie/pytania różnicujące \
→ rozpoznanie wstępne → zalecenia.
6. Rozpoznanie (`expected_diagnosis`) musi pasować do objawów.
7. Jeśli to możliwe, podaj kod ICD-10 w `icd10_code`.

Zasady dot. danych osobowych (ważne — to dataset do trenowania \
anonimizatora):
- `patient_full_name`: użyj realistycznego polskiego imienia i nazwiska \
(FIKCYJNEGO, nie znanej osoby publicznej).
- `patient_pesel`: wygeneruj ciąg 11 cyfr w formacie PESEL (może być \
niepoprawny matematycznie — chodzi o wzorzec).
- W treści dialogu wpleć naturalnie: imię pacjenta, nazwisko, \
czasem wiek, adres lub miasto, numer telefonu. Te dane będą celem \
treningowym dla modułu anonimizacji — muszą być obecne w tekście, nie \
tylko w polach metadanych.
- NIE używaj danych prawdziwych osób.

Zasady techniczne:
- Wszystkie pola tekstowe po polsku (nazwy enumów pozostają po angielsku: \
`doctor`, `patient`).
- Pole `model` pozostaw puste ("") — zostanie uzupełnione przez system.
- Pole `generated_at` pozostaw puste — zostanie uzupełnione przez system.
"""


USER_PROMPT_TEMPLATE = """\
Wygeneruj JEDEN scenariusz wizyty lekarskiej dla specjalizacji: \
**{specialization}**.

Wymagania:
- Liczba tur dialogu: od {min_turns} do {max_turns}.
- Przypadek kliniczny typowy dla tej specjalizacji, ale zróżnicuj objawy \
i stopień ciężkości.
- Pamiętaj o wplecieniu w treść dialogu danych osobowych pacjenta \
(imię, nazwisko, czasem miasto/telefon) jako celu dla anonimizatora.

Zwróć wyłącznie JSON zgodny ze schematem ScenarioModel.
"""


def render_user_prompt(
    specialization: Specialization,
    min_turns: int,
    max_turns: int,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        specialization=specialization.value,
        min_turns=min_turns,
        max_turns=max_turns,
    )


def _example_kardiologia() -> ScenarioModel:
    return ScenarioModel(
        id="example-kardio-01",
        specialization=Specialization.kardiologia,
        patient_age=62,
        patient_sex="M",
        patient_full_name="Jan Kowalski",
        patient_pesel="62050512345",
        chief_complaint="Ucisk w klatce piersiowej przy wysiłku",
        expected_diagnosis="Stabilna choroba wieńcowa",
        icd10_code="I25.1",
        turns=[
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Dzień dobry, panie Janie. Proszę usiąść. Co pana sprowadza?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Dzień dobry, panie doktorze. Od jakiegoś miesiąca, jak "
                    "wchodzę po schodach na drugie piętro, to mnie tak ściska "
                    "w klatce, za mostkiem."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "Czy ten ból promieniuje gdzieś — do ręki, żuchwy, pleców?"
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Tak, czasem do lewej ręki. I się pocę wtedy.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="A ustępuje w spoczynku? Po ilu minutach?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Jak przystanę, to po dwóch, trzech minutach przechodzi.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "To brzmi jak dławica piersiowa wysiłkowa. Skierujemy pana "
                    "na próbę wysiłkową i echo serca. Proszę tu podać adres — "
                    "w Krakowie pan mieszka?"
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Tak, ulica Długa 14, telefon 512 334 221.",
            ),
        ],
        model="example",
        seed=None,
    )


def _example_pulmonologia() -> ScenarioModel:
    return ScenarioModel(
        id="example-pulmo-01",
        specialization=Specialization.pulmonologia,
        patient_age=34,
        patient_sex="F",
        patient_full_name="Anna Nowak",
        patient_pesel="90081567823",
        chief_complaint="Duszność i świsty przy oddychaniu",
        expected_diagnosis="Zaostrzenie astmy oskrzelowej",
        icd10_code="J45.9",
        turns=[
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Dzień dobry, pani Anno. Jak się pani dziś czuje?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Słabo. Od tygodnia ciężko mi oddychać, zwłaszcza w nocy. "
                    "Świszczy mi w płucach."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Używa pani salbutamolu? Ile razy dziennie ostatnio?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Cztery, pięć razy. Wcześniej wystarczał raz na tydzień.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Czy był kontakt z alergenami? Kot, pyłki, pleśń w domu?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Przyjechałam do rodziców pod Warszawę, mają kota. Może od "
                    "tego."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "To prawdopodobna przyczyna. Zrobimy spirometrię i "
                    "zmodyfikujemy leczenie kontrolne."
                ),
            ),
        ],
        model="example",
        seed=None,
    )


def _example_medycyna_rodzinna() -> ScenarioModel:
    return ScenarioModel(
        id="example-mr-01",
        specialization=Specialization.medycyna_rodzinna,
        patient_age=28,
        patient_sex="M",
        patient_full_name="Piotr Wiśniewski",
        patient_pesel="95112034512",
        chief_complaint="Gorączka i ból gardła",
        expected_diagnosis="Ostre zapalenie gardła",
        icd10_code="J02.9",
        turns=[
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Dzień dobry, panie Piotrze. Co dolega?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Od trzech dni mam gorączkę, 38.5, i strasznie boli "
                    "gardło przy przełykaniu."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Kaszel, katar? Ktoś w domu chory?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Bez kaszlu. Syn chodzi do przedszkola, tam teraz chorują.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "Zajrzę panu w gardło. Migdałki zaczerwienione, z nalotem. "
                    "Zrobimy szybki test na paciorkowca."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Dobrze.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "Test dodatni. Przepiszę fenoksymetylopenicylinę na 10 dni. "
                    "Receptę wyślę na telefon 608 112 445, tak?"
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Tak, dziękuję panie doktorze.",
            ),
        ],
        model="example",
        seed=None,
    )


def _example_endokrynologia() -> ScenarioModel:
    return ScenarioModel(
        id="example-endo-01",
        specialization=Specialization.endokrynologia,
        patient_age=51,
        patient_sex="F",
        patient_full_name="Maria Zielińska",
        patient_pesel="73030489912",
        chief_complaint="Zmęczenie, przyrost wagi, wypadanie włosów",
        expected_diagnosis="Niedoczynność tarczycy (choroba Hashimoto)",
        icd10_code="E06.3",
        turns=[
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Dzień dobry pani Mario. Co panią sprowadza?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Ciągle jestem zmęczona, przytyłam sześć kilogramów w pół "
                    "roku, włosy mi wypadają. I jest mi zimno nawet latem."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Ma pani jakieś wyniki tarczycy ostatnio?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Tak, TSH 8.2, wolne T4 poniżej normy.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "To jawna niedoczynność. Zlecę anty-TPO, żeby potwierdzić "
                    "Hashimoto, i USG tarczycy."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="A leczenie?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "Lewotyroksyna 50 mikrogramów rano. Kontrola TSH za 6-8 "
                    "tygodni. Pani mieszka w Gdańsku, prawda?"
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Tak, ul. Wrzeszczańska 7.",
            ),
        ],
        model="example",
        seed=None,
    )


def _example_neurologia() -> ScenarioModel:
    return ScenarioModel(
        id="example-neuro-01",
        specialization=Specialization.neurologia,
        patient_age=45,
        patient_sex="F",
        patient_full_name="Katarzyna Lewandowska",
        patient_pesel="79062214321",
        chief_complaint="Napady pulsującego bólu głowy z aurą",
        expected_diagnosis="Migrena z aurą",
        icd10_code="G43.1",
        turns=[
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Dzień dobry pani Katarzyno. Proszę opisać, co się dzieje.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Od pół roku mam napady silnego bólu głowy, tylko po "
                    "prawej stronie. Pulsuje mi w skroni przez kilka godzin."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Widzi pani coś dziwnego przed bólem? Migające światła?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text=(
                    "Tak, takie błyski w lewym oku, mniej więcej pół godziny "
                    "przed."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text="Nudności, wrażliwość na światło, dźwięk?",
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="Wszystko. Muszę się położyć w ciemnym pokoju.",
            ),
            DialogueTurn(
                speaker=SpeakerRole.doctor,
                text=(
                    "To klasyczny obraz migreny z aurą. Zrobimy MRI głowy "
                    "kontrolnie. Proszę telefon do kontaktu."
                ),
            ),
            DialogueTurn(
                speaker=SpeakerRole.patient,
                text="601 445 112. Mieszkam w Poznaniu.",
            ),
        ],
        model="example",
        seed=None,
    )


FEW_SHOT_EXAMPLES: dict[Specialization, ScenarioModel] = {
    Specialization.kardiologia: _example_kardiologia(),
    Specialization.pulmonologia: _example_pulmonologia(),
    Specialization.medycyna_rodzinna: _example_medycyna_rodzinna(),
    Specialization.endokrynologia: _example_endokrynologia(),
    Specialization.neurologia: _example_neurologia(),
}


def render_few_shot_block(specialization: Specialization) -> str:
    """Renders the hand-written example for a given specialization as JSON.

    Inserted into the user prompt to anchor format and tone.
    """
    example = FEW_SHOT_EXAMPLES[specialization]
    return example.model_dump_json(indent=2, exclude={"generated_at"})
