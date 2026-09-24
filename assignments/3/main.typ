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
#let argmax = $op("argmax", limits: #true)$
#let sgn = $op("sgn", limits: #true)$
#pagebreak()

// =====================================================================

= Optimal Classfication (16 points)

+ From the lecture notes we know that if class posteriors are given the optimal classifier is $hat(y) = argmax_k (Y=C_k | X=x)$.
  We can define the function $g = cases(1 "if" x>0, 0 "if" x<=0)$. In this case the optimal classifier is
  $ hat(y)(x) = g(P(Y=1 |X = x) - P(Y=0 | X=x)) $

  We have $hat(y)(0) = g(0 - 1) = 0$ and $hat(y)(1) = g(0.2 - 0.8) = 0$, so
  $ hat(y)(x) = 0 $
  And also $P("error" | X = 1) = 1 - P(Y = hat(y)(1) | X =1 ) = 0.2$ and $P("error" | X = 0) = 1 - P(Y= hat(y)(0) | X = 0) = 0$.
  Since all inputs occur with equal probablility $P(X=x) = 0.5$. The risk is therefore
  $ R(hat(y)) = sum_(x in {0,1}) P(X=x) * P("error" | X=x) = 0.5 * (0.2 + 0)= 0.1 $.
+ If $x = 0$  the probabilistic classifier is always correct, since it always correctly predicts the label $y=0$, i.e. $P("error" | X = 0) = 0$. For $x=0$ if the true label is $y=1$, which happens 80% of times, the classifier errors 20% of times. For $x=1$ if the true label is $y=1$, which happens 20% of times, the classifier errors 80% of times. So $P("error" | X = 1) = 0.8 * 0.2 + 0.2 * 0.8 = 0.32$. The risk therefore is
$ R = 0.5 (0 + 0.32) = 0.16 $

= Logistic Regression (50 points)

== Cross-entropy error measure (16 points)

=== Rewriting negative logarithmic likelihood as cross-entropy
We have that
$ h(x) = P( Y=1 | X =x) = 1 /(1 + e^(-w^t x)) =(e^(w^t x))/(1+e^(w^t x)) $
Thus
$
  cal(L)(w) & = 1/N sum_(n=1)^(N) [bb(1)_(y_n=1) ln 1 / h(x_n) + bb(1)_(y=-1) ln 1/(1 - h(x_n))] \
            & = 1/N sum_(n=1)^(N) [bb(1)_(y_n=1) ln (1+e^(-w^t x_n)) + bb(1)_(y_n=-1) ln (1+e^(w^t x_n))] \
            & = 1/N sum_(n=1)^(N) [ln (1+e^(-y_n w^t x_n))]
$

=== Cross-entropy and logistic regression

== Logistic Regression loss gradient (24 points)

=== Gradient for -1 and 1
We have
$
  gradient cal(L)(w) & = gradient 1/N sum_(n=1)^N ln (1+ e^(-y_n w^t x_n)) \
                     & = 1/N sum_(n=1)^N -1/(1+ e^(-y_n w^t x_n)) y_n x_n e^(-y_n w^t x_n) \
                     & = 1/N sum_(n=1)^N -1/(1+ e^(y_n w^t x_n)) y_n x_n \
                     & = 1/N sum_(n=1)^N - y_n x_n theta(-y_n w^t x_n) \
$
=== Gradient for 0 and 1

To get the label space {0,1} we can substitute $y_n$ with $t= (y_n + 1)/2$, where we get $g=-1/N sum^N_(n=1) (t x_n)/(1+e^(t w^t x_n))$. When we plug in -1 and 1 for $y_n$ in $g$ and in $g'=-1/N sum_(n=1)^N [y_n - theta (w^t x_n)]$, we see that they're equal.

+ For $y_n=1$ we have
  - For $g$: $-1/N sum^N_(n=1) (x_n)/(1+e^(w^t x_n)) = -1/N sum^N_(n=1) 1 - theta(w^t x_n)$
  - For $g'$: $-1/N sum^N_(n=1) 1 - theta(w^t x_n)$
...

Therefore they're both equal in both cases.


=== Influence of misclassified examples

== Log-odds
Let me define $p = P(Y=1| X=x)$. We have that $sigma$ as the logistic function satisfies the equation:

$
  sigma (w^T x + b) & = sigma (ln (P(Y=1| X=x))/P(Y=0 | X = x )) \
                    & = sigma (ln p/(1-p)) \
                    & = 1/(1+e^(-ln p/(1-p))) \
                    & = 1/(1+ (1-p)/p) \
                    & = 1/((p+ (1-p))/p) \
                    & = p = P(Y=1| X=x)
$

= Sleep Well (34 points)

== Data understanding and preprocessing

The frequencies of the class labels was calculated using numpy and is displayed in @classfreq.

#figure(
  table(
    columns: 6,
    align: center,
    [Class], [0], [1], [2], [3], [4],
    [Frequency], [0.52], [0.10], [0.25], [0.05], [0.08],
  ),
  caption: "Class frequencies",
)<classfreq>

== Classification

=== Logistic Regression
The sklearn library was used to compute the regression. No regularization was used. Errors are shown in @lr.

#figure(
  table(
    columns: 2,
    align: center,
    [Test Loss], [Training Loss],
    [0.10], [0.15],
  ),
  caption: "Logistic Regression Error",
)<lr>


=== Random Forest
The sklearn library was used to compute the random forests. Errors are shown in @rf. No regularization was used.

#figure(
  table(
    columns: 4,
    align: center,
    [Number of Trees], [Test Loss], [Training Loss], [OOB Score],
    [50], [0.11], [0], [0.8476],
    [100], [0.11], [0], [0.8501],
    [200], [0.11], [0], [0.8518],
  ),
  caption: "Random Forest Error",
)<rf>

=== Nearest Neighbor
The sklearn library was used to compute the KNN classification. Errors are shown in @knn. No regularization was used. The number of neighbors was determined by doing cross validation for every $k in {1, ..., 49}$ and picking the $k$ which had the least error.

#figure(
  table(
    columns: 3,
    align: center,
    [Test Loss], [Training Loss], [K],
    [0.10], [0.15], [46],
  ),
  caption: "KNN Error",
)<knn>
