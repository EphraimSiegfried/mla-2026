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

#pagebreak()
= Digits Classification with $K$ Nearest Neighbors (40 points)

== Task \#1

The K Nearest Neighbors algorithm is shown in @knn.

#figure(
  ```python
  def knn(
      training_points: np.ndarray,
      training_labels: np.ndarray,
      test_points: np.ndarray,
      test_labels: np.ndarray,
  ):
      _, m = np.shape(training_points)
      _, n = np.shape(test_points)
      distances = (
          np.outer(np.diagonal(training_points.T @ training_points), np.ones(n))
          - 2 * training_points.T @ test_points
          + np.outer(np.ones(m), np.diagonal(test_points.T @ test_points))
      )
      distances_argsort = np.argsort(distances, axis=0)

      labels_ext = np.tile(training_labels, (n, 1)).T
      labels_sorted = np.take_along_axis(labels_ext, distances_argsort, axis=0)
      predictions = np.cumsum(labels_sorted, axis=0)
      predictions = np.where(predictions > 0, 1, -1)

      error = predictions != test_labels
      avg_error = np.average(error, axis=1)
      return avg_error
  ```,
  caption: [K Nearest Neighbors Algorithm],
)<knn>

The validation error for each of the validation sets is depicted below. The variance over the validation sets is shown in @knn-variance. We can observe that the fluctuations decreases with increasing $n$. This is clearly visible in @knn-variance, where for example for $n=80$ the variance is lower than for $n=10$. Over all validation sets we observe that the accuracy of the predictor decreases when $K$ gets larger.


#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("figures/knn-error-rate-n=10.png"),
    caption: [n = 10],
  ),
  figure(
    image("figures/knn-error-rate-n=20.png"),
    caption: [n = 20],
  ),

  figure(
    image("figures/knn-error-rate-n=40.png"),
    caption: [n = 40],
  ),
  figure(
    image("figures/knn-error-rate-n=80.png"),
    caption: [n = 80],
  ),

  [
    #figure(
      image("figures/knn-error-variance.png"),
      caption: [n = 80],
    ) <knn-variance>
  ],
)

#pagebreak()
= Regression (50 points)

== Task 1 (5 points)

The linear regression algorithm is depicted in @regression.

#figure(
  ```python
  def linear_regression(x: np.ndarray, y: np.ndarray):
      n = np.shape(x)[0]
      x = np.c_[x, np.ones(n)]
      wt = np.linalg.solve(x.T @ x, x.T @ y)
      w = wt[:-1]
      b = wt[-1]
      return w, b
  ```,
  caption: [K Nearest Neighbors Algorithm],
)<regression>


== Task 2 (10 points)

The parameters of the model h are $w=0.26$ and $b=0.032$. The mean squared error is 34.83.

== Task 3 (10 points)


== Task 4 (5 points)
The regression with the model $h=exp(a x + b)$ is visible in @regression-h1.
#figure(
  image("figures/regression-h1.png", width: 80%),
  caption: [n = 80],
) <regression-h1>

== Task 5 (10 points)

The coefficient of determination is $R^2=0.36$. We have that $R^2$ is 1 if $sum_(i=1)^N (y_i - h(x_i))^2$ is zero. In this case the model prediction perfectly aligns with the data. If $R^2=0$ then the numerator and denominator have to be equal which means that the model performs equally well as just using the mean as prediction. $R^2$ can be negative if the predictions are worse than guessing with the mean. So overall $R^2$ measures how much better than the mean the model performs.

== Task 6 (10 points)
The regression with the model $h′=exp(a sqrt(x) + b)$ is visible in @regression-h1. The mean squared error is 28.08 and $R^2=0.48$. Since the MSE is lower and $R^2$ is higher this model performs better than the previous model.

#figure(
  image("figures/regression-h2.png", width: 80%),
  caption: [n = 80],
) <regression-h2>
