# Praca magisterska — MedVoicePL (LaTeX)

Szablon pracy magisterskiej PJATK przepisany z Typst na **LaTeX + XeLaTeX**.
Tekst pracy w języku polskim. Projekt: *MedVoicePL — multimodalny,
wieloagentowy system hybrydowy do wspomagania decyzji klinicznych (CDSS)*.

## Struktura

```
Praca_magisterska/
├── main.tex                 # root, \input-uje wszystko inne
├── thesis-config.tex        # metadane (autor, promotor, tytuł)
├── preamble/
│   ├── packages.tex         # \usepackage{...}
│   ├── pjatk-style.tex      # fontspec EB Garamond, titlesec, listings
│   ├── titlepage.tex        # strona tytułowa PJATK
│   └── abstract.tex         # streszczenie + słowa kluczowe
├── chapters/
│   ├── 01-wstep.tex
│   ├── 02-stan-wiedzy.tex
│   ├── 03-wymagania.tex
│   ├── 04-architektura.tex
│   ├── 05-implementacja.tex
│   ├── 06-ewaluacja.tex
│   └── 07-wnioski.tex
├── bibliography/
│   └── references.bib       # BibLaTeX (biber)
└── contents/assets/
    ├── fonts/EBGaramond-*.ttf   # 10 plików, ładowane przez fontspec
    └── images/PJATK_pl_poziom_1.pdf   # logo PJATK
```

## Wymagania

- **TeX Live 2023+** lub **MacTeX 2023+** (w pełnej instalacji — `scheme-full`).
  W skróconej wersji (`scheme-basic`) trzeba doinstalować: `fontspec`,
  `babel-polish`, `biblatex`, `biber`, `listings`, `titlesec`, `microtype`,
  `fancyhdr`, `enumitem`, `csquotes`, `hyperref`.
- **Kompilator:** XeLaTeX (czyta TTF wprost z `contents/assets/fonts/`, bez
  instalacji czcionki w systemie).
- **Backend bibliografii:** Biber (nie BibTeX).
- **VS Code** + rozszerzenie [LaTeX Workshop][1].

[1]: https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop

## Kompilacja z terminala

```bash
cd Praca_magisterska
xelatex -interaction=nonstopmode main
biber main
xelatex -interaction=nonstopmode main
xelatex -interaction=nonstopmode main
```

Cztery kroki są potrzebne, aby odświeżyć spis treści, listę rysunków/tabel/
listingów, bibliografię i odnośniki.

Szybciej: `latexmk -xelatex main.tex` (latexmk sam zapętli kompilacje).

## LaTeX Workshop — konfiguracja VS Code

Pierwsze wiersze `main.tex` (`% !TEX program = xelatex` i
`% !BIB program = biber`) informują LaTeX Workshop, jakiego kompilatora
używać. W większości przypadków to wystarczy, żeby build ruszył.

Jeżeli build mimo to nie startuje z domyślnej instalacji, dodaj do
`.vscode/settings.json` (w tym katalogu) lub do globalnych ustawień VS Code:

```jsonc
{
  "latex-workshop.latex.recipe.default": "xelatex + biber",
  "latex-workshop.latex.recipes": [
    {
      "name": "xelatex + biber",
      "tools": ["xelatex", "biber", "xelatex", "xelatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    { "name": "xelatex", "command": "xelatex",
      "args": ["-synctex=1", "-interaction=nonstopmode",
               "-file-line-error", "%DOC%"] },
    { "name": "biber", "command": "biber", "args": ["%DOCFILE%"] }
  ],
  "latex-workshop.view.pdf.viewer": "tab"
}
```

Build: `Ctrl/Cmd + Alt + B`. Podgląd PDF: `Ctrl/Cmd + Alt + V`.

## Przepływ pracy

1. Edytuj `thesis-config.tex` — tytuł, autor, promotor, specjalizacja,
   numer albumu (`sXXXXX` → prawdziwy numer indeksu).
2. Edytuj `preamble/abstract.tex` — streszczenie i słowa kluczowe.
3. Pisz rozdziały w `chapters/*.tex`. Każdy ma etykietę `\label{ch:...}`
   do odwoływania się (`\ref{ch:wstep}`).
4. Dodawaj źródła do `bibliography/references.bib` i cytuj przez
   `\cite{klucz}` lub `\parencite{klucz}`.
5. Rysunki wkładaj do `contents/assets/images/` i używaj
   `\includegraphics{nazwa}` (katalog jest już w `\graphicspath`).

## Co zostało zmienione względem szablonu Typst

Oryginalny szablon `pjatk-template.typ` był dwujęzyczny (PL/EN). Ta wersja
jest **tylko polska** — jedna strona tytułowa, streszczenie tylko po polsku.
Jeżeli kiedyś będziesz potrzebować wersji angielskiej, trzeba rozszerzyć
`titlepage.tex` i `abstract.tex` o drugi zestaw sekcji, analogicznie do
oryginalnego `commonPhrases`.

Zamiennik pakietu `codly` (syntax highlighting w Typst) to
[`listings`](https://ctan.org/pkg/listings) skonfigurowany w
`preamble/pjatk-style.tex` (numeracja po prawej, kolorowanie składni,
łamanie linii). Dla zaawansowanego highlightingu warto rozważyć
[`minted`](https://ctan.org/pkg/minted), ale wymaga on Pythona i
`--shell-escape` przy kompilacji.

## Licencja

Treść pracy: © Jakub Boczar, 2026.
Szablon: zachowuje konwencje PJATK (Polsko-Japońska Akademia Technik Komputerowych).
Czcionka EB Garamond: SIL Open Font License 1.1.
