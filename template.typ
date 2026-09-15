#set document(title: "Machine Learning A (2026) — Home Assignment 1", author: "Ephraim Siegfried (hsc282)")

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  numbering: "1",
)

// 12pt article base font
#set text(font: "New Computer Modern", size: 12pt, lang: "en")
#set par(justify: true, leading: 0.65em, first-line-indent: 1.5em)

// article.cls sectioning: \Large\bfseries / \large\bfseries, numbered 1, 1.1
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 17.28pt)
#show heading.where(level: 2): set text(size: 14.4pt)
#show heading.where(level: 3): set text(size: 12pt)
#show heading: set block(above: 1.4em, below: 0.9em)

#set enum(indent: 1em, spacing: 0.9em)
#set list(indent: 1em, spacing: 0.9em)

// hyperref-ish: make internal references clickable and coloured
#show ref: set text(fill: rgb("#005CC5"))
#show link: set text(fill: rgb("#005CC5"))

// TOC: article.cls bolds section entries (number, title and page number),
// and gives them no dot leader. Subsections stay upright with dots.
#show outline.entry.where(level: 1): it => strong(it)

// ---------------------------------------------------------------------
// Code listing style (the "mystyle" lstdefinestyle equivalent)
// ---------------------------------------------------------------------
#set raw(tab-size: 4)

#show raw.where(block: true): it => block(
  width: 100%,
  fill: rgb("#F7F7F7"),
  stroke: 0.6pt + rgb("#EEEEEE"),
  inset: 8pt,
  breakable: true,
  text(size: 10pt, it),
)

#show raw.where(block: false): it => box(
  fill: rgb("#F7F7F7"),
  inset: (x: 2pt),
  outset: (y: 3pt),
  text(size: 10pt, it),
)

// ---------------------------------------------------------------------
// \maketitle
// ---------------------------------------------------------------------
#align(center)[
  #v(2em)
  #text(size: 20.74pt)[
    Machine Learning A (2026) \
    Home Assignment 1
  ]
  #v(1.5em)
  #text(size: 14.4pt, fill: red)[Ephraim Siegfried (hsc282)]
  #v(2em)
]

// Please leave the table of contents as is, for the ease of navigation for TAs
#outline(title: "Contents", depth: 3, indent: auto)

#pagebreak()

// =====================================================================

= Make Your Own (10 points)

= Digits Classification with $K$ Nearest Neighbors (40 points)

== Task \#1

== Task \#2 (optional, not for submission)

= Regression (50 points)

== Task 1 (5 points)

== Task 2 (10 points)

== Task 3 (10 points)

== Task 4 (5 points)

== Task 5 (10 points)

== Task 6 (10 points)
