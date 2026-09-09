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

+ As features I would choose their average university grade and their estimation of how many hours per week they intend to study. So the sample space is $cal(X) = RR times RR$.

+ The label space contains final machine learning grades $cal(Y) = {12, 10, 7, 4, 02, 00, -3}$.

+ The 0-1 loss is not suitable in this case because it doesn't capture how severe a mistake is. Since the magnitude of the error is meaningful I would choose the square loss function $cal(l)(Y', Y) = (Y'-Y)^2$.

+ For the distance measure I would choose the euclidian distance $||x -x'|| = sqrt((x-x')^T (x-x'))$ .

+ I would split the sample data $S$ into a training set $S^(t r a i n)$ and in a validation set $S^(v a l)$. I would train my classifier with the training set and validate the performance with the validation set by evaluating the mean squared loss of it.

+ Issues that could come up is that students lie about their self estimate of how many hours they intend to study in order to get a better grade (by overestimation). This issue could be alleviated by somehow counting the hours they study per week.

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

// --- Placeholder figure ----------------------------------------------
#figure(
  // Replace this rectangle with:  image("plot.png", width: 50%)
  rect(width: 50%, height: 4cm, fill: luma(240), stroke: 0.5pt)[
    #align(center + horizon)[example-image]
  ],
  caption: [#lorem(35)],
) <fig-placeholder>

// --- Placeholder code -------------------------------------------------
```python
# Example code
# Creating an example array
data = np.array([5, 2, 8, 1, 6])
# Calculating cumulative sum using cumsum
cumulative_sum = np.cumsum(data)
```
