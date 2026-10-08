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
    Home Assignment 5
  ]
  #v(1.5em)
  #text(size: 14.4pt, fill: red)[Ephraim Siegfried (hsc282)]
  #v(2em)
]

// Please leave the table of contents as is, for the ease of navigation for TAs
#outline(title: "Contents", depth: 3, indent: auto)
#let argmax = $op("argmax", limits: #true)$
#let sgn = $op("sgn", limits: #true)$
#pagebreak()

// =====================================================================

= On the Role of Dependence

Let $X = X_1 = X_2 = ... = X_n$ and let $mu = EE[X_1] = 1/2$. Then
$
  PP(|mu - 1/n sum_i=1^n X_i | >= 1/2) =
  PP(|mu - 1/n n X | >= 1/2) =
  PP(|mu - X | >= 1/2)
$

We have that in all possible outcomes of $X$:
$
  PP(|mu - 1 | >= 1/2) = PP(|1/2-1| >= 1/2) = PP(1/2 >= 1/2) = 1
$
and
$
  PP(|1/2 - 0 | >= 1/2) = PP(1/2 >= 1/2) = 1
$

= On Confidence Intervals

== Part 1
+ False. The confidence interval only states that the mean lies within that range with a 98% confidence, but a mean in that range can be constructed with observations outside this range.
+ False. Same reason as above.
+ False. Because the interval is computed using the samples (which are drawn randomly from a distribution) the interval is random itself. Therefore another set of observations will not necessary create the same interval.
+ True. If using the same sample, then $J subset.eq J'$. A higher confidence means that the interval is more broad and a lower confidence interval computed with the same method and same observation just creates a subinterval.

== Part 2

The interval would be $I'= [0.24, 0.44]$. This broader interval increases the confidence that $theta$ is in the inteval, so it ensures at least the same confidence degree. It is also the tightest such interval.


= Loss Range Correction in Generalization Bounds

#set enum(numbering: "(i)")

+ No because Hoeffding's inequality depends on the support of the random variables, in this case the loss function. A [0.18, 1.44] range leads to a different inequality than the [0, 1] range. So therefore the reported bound is invalid.
+ Hoeffding's inequality for generic ranges is $PP(S_n-EE[S_n] >= S_n) <= e^(-(2epsilon^2)/(sum^n_(i=1) (b_i - a_i)^2))$. Let's denote $z := 1.44-0.18$ and $Z_i := cal(l)(h(X_i), Y_i)$. We have
$
  PP(L(h) - hat(L)(h, S) >= epsilon) = PP(EE[1/n sum^n_(i=1) Z_i] - 1/n sum^n_(i=1) Z_i) <= e^(-(2epsilon^2)/(sum^n_(i=1) (b_i - a_i)^2)) = e^(-(2epsilon^2)/(n z^2))
$

Setting $delta = e^(-(2epsilon^2)/(n z^2))$ yields $epsilon = sqrt((n z^2)/2 ln(1/d))$. We have the equations

#set math.equation(numbering: "(1)")

$ hat(L)(h,S) + sqrt(1/(2 n) log(1/d)) = 0.38 $ <eq-1>
$ hat(L)(h,S) + sqrt((n z^2)/2 ln(1/d)) = x $ <eq-2>

To solve for x we subtract (2) - (1)

$
  x = sqrt((n z^2)/2 ln(1/d)) - sqrt(1/(2 n) log(1/d)) + 0.38 =
$

= Convolutional Neural Networks

The $G_x$ convolution is displayed in @fig-sobel-gx, the $G_y$ convolution is displayed in @fig-sobel-gy and the final feature map can be seen in @fig-sobel-g. The difference between convolution and cross correlation is that in cross correlation the kernel is flipped by $180 degree$. This would mean that in $G_x$ and $G_y$ that the black lines would be white and vice versa.

#grid(
  columns: 3,
  gutter: 1em,
  [ #figure(image("figures/sobel_gx.jpg"), caption: [$G_x$]) <fig-sobel-gx> ],
  [ #figure(image("figures/sobel_gy.jpg"), caption: [$G_y$]) <fig-sobel-gy> ],
  [ #figure(image("figures/sobel_g.jpg"), caption: [$G$]) <fig-sobel-g> ],
)
