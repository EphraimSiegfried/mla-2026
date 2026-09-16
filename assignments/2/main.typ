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

= Preprocessing (35 points)

== Importance of Preprocessing (8 points)

Whether BoL should give person C credit depends on the distance function used. If we assume it's the euclidean distance, then they should give him credit since the distance between them and A is $sqrt((21 - 47)^2 + (36000 - 35000)^2) approx 1000$ which is smaller than the distance between them and B, that is $sqrt((21 - 22)^2 + (36000 - 40000)^2) approx 4000$. But if income is measured in dollars then they should not lend C money since the distance between them and A is $sqrt((21 - 47)^2 + (36 - 35)^2) approx 26$ which is larger than the distance to B $sqrt((21 - 22)^2 + (36 - 40)^2) approx 4$.
== Input Centering (9 points)

#set enum(numbering: "(a)")
+ We have that
  $
    gamma X & = (bold(I) - 1/N bold(1)bold(1)^T)X \
            & = X - 1/N bold(1)bold(1)^T X \
            & = X - bold(1) 1/N bold(1)^T X \
            & = X - bold(1) (1/N X^T bold(1))^T \
            & = X - bold(1) overline(x) = Z
  $
+ TODO

== Input Whitening (18 points)
+ #set enum(numbering: "i.")
  + $"Var"(x_1) = "Var"(hat(x)_1) = 1$ #linebreak()
  + $"Var"(x_2) &= "Var"(sqrt(1-epsilon^2)hat(x)_1 + epsilon hat(x)_2) \
    &= "Var"(sqrt(1-epsilon^2)hat(x)_1) + "Var"(epsilon hat(x)_2) \
    &= (1-epsilon^2)"Var"(hat(x)_1) +epsilon^2"Var"(hat(x)_2) \
    &= 1 - epsilon^2 + epsilon^2 = 1$
  + $"Cov"(x_1, x_2) &= 1/2("Var"(x_1 + x_2) - "Var"(x_1) - "Var"(x_2)) \
    &= 1/2 ("Var"(x_1 + x_2) - 2) \
    &= 1/2 ("Var"(hat(x)_1 + sqrt(1-epsilon^2)hat(x)_1 + epsilon hat(x)_2) - 2) \
    &= 1/2 ("Var"((1+sqrt(1-epsilon^2))hat(x)_1 + epsilon hat(x)_2) - 2) \
    &= 1/2 ((1+sqrt(1-epsilon^2))^2"Var"(hat(x)_1) + epsilon^2"Var"(hat(x)_2)) - 2) \
    &= 1/2 ((1+sqrt(1-epsilon^2))^2 + epsilon^2) - 2) \
    &= 1/2 ((1 + 2sqrt(1- epsilon^2) + 1-epsilon^2 + epsilon^2) - 2) = sqrt(1-epsilon^2)$
+ We know that $hat(x)_1 = x$ and $x_2 = sqrt(1-epsilon^2)hat(x)_1 + epsilon hat(x)_2 <=> hat(x)_2=(x_2-sqrt(1-epsilon^2)x_1)/epsilon$. We can substitute and find that
  $
    f(x) & = hat(w)_1 x_1 + hat(w)_2 (x_2-sqrt(1-epsilon^2)x_1)/epsilon \
         & = x_1((hat(w)_1 epsilon - hat(w)_2 sqrt(1-epsilon^2))/epsilon) + x_2(hat(w)/epsilon)
  $
  Therefore $w_1, w_2$ can be expressed as
  $
    w_1 & = (hat(w)_1 epsilon - hat(w)_2 sqrt(1-epsilon^2))/epsilon \
    w_2 & = hat(w)/epsilon
  $

= Hoeffding's Bound (15 points)
The expected value of the random variable $X$ is $EE[X] = 1/3 (-2 + 0.8 + 1) = -0.2/3$. The expected value of S is $EE[S] = EE[sum^100_(i=1) X_i] = sum^100_(i=1) EE[X_i] = 100 * -0.2/3 = -20/3$. The lower bound of $X$ is $a=-2$ and upper bound is $b=1$. Choose $epsilon = 86/3$ and with Hoeffding's Bound we have
$
  PP(S_n - EE[S_n] >= epsilon) & = PP(S_n + 20/3 >= 86/3) \
                               & = PP(S_n >= 22) <= e^(- (2 epsilon^2) / (sum_(i=1)^n (b_i - a_i)^2)) \
                               & = PP(S_n >= 22) <= e^(- (2 * (86/3)^2) / (100 * 3^2)) approx 0.161 \
$

= Illustration of Markov's, Chebyshev's, and Hoeffding's Inequalities (25 points)

#set enum(numbering: "3.a")
+ #set enum(numbering: "1.")
  +



