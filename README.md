# inferenceql.gpm.glm

![tests](https://github.com/probcomp/inferenceql.gpm.glm/workflows/tests/badge.svg)
![linter](https://github.com/probcomp/inferenceql.gpm.glm/workflows/linter/badge.svg)

Linear models.

## Prerequisites

Model training requires [OpenBLAS](https://www.openblas.net/).

``` shell
brew install openblas
```

## Usage

This library is intended for use with [inferenceql.query](https://github.com/probcomp/inferenceql.query). First, launch a REPL with inferenceql.query on the class path:

``` shell
clj -Sdeps '{:deps {probcomp/inferenceql.query {:git/url "git@github.com:probcomp/inferenceql.query.git" :sha "…"}}}'
```

Next, define the model and the data set to be queried. `model->gpm` accepts instances of `smile.regression.LinearModel` or instances of `smile.glm.GLM`:

``` clojure
(require '[clojure.walk :as walk]
         '[inferenceql.gpm.glm.smile :as smile])

(import '[smile.data.formula Formula]
        '[smile.io Read]
        '[smile.regression OLS])

(def data-frame (Read/arff "https://raw.githubusercontent.com/meteoinfo/MIML/master/miml/datasets/data/weka/regression/2dplanes.arff"))

(def data
  (-> data-frame
      (smile/data-frame->maps)
      (walk/keywordize-keys)))

(def smile-model (OLS/fit (Formula/lhs "y") data-frame))

(def gpm (smile/model->gpm smile-model "y" (.names data-frame)))
```

``` clojure
(import '[smile.data.formula Formula]
        '[smile.glm GLM]
        '[smile.glm.model Bernoulli])

(def data
   (let [names [:x :y :z]
         gen-cell #(rand-int 2)
         gen-row #(repeatedly (count names) gen-cell)
         gen-dataset #(repeatedly 20 gen-row)]
     (smile/vectors->data-frame Integer/TYPE names (gen-dataset))))

(def model (GLM/fit (Formula/lhs "x") data (Bernoulli/logit)))
```

Finally, launch the InferenceQL REPL:

``` clojure
(require '[inferenceql.query.main :as main])

(main/repl data {:model gpm})
```

``` sql
select * from generate y under model conditioned by x1=1 and x2=1 and x3=1 and x4=1 and x5=1 and x6=1 and x7=1 and x8=1 and x9=1 and x10=1 limit 5;
```
