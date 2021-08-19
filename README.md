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

Next, define the model and the data set to be queried:

``` clojure
(require '[clojure.walk :as walk]
         '[inferenceql.gpm.glm.smile :as smile]
         '[inferenceql.query.main :as main])

(import '[smile.io Read]
        '[smile.regression OLS]
        '[smile.data.formula Formula])

(def data-frame (Read/arff "https://raw.githubusercontent.com/meteoinfo/MIML/master/miml/datasets/data/weka/regression/2dplanes.arff"))

(def data (-> data-frame
              (smile/data-frame->maps)
              (walk/keywordize-keys)))

(def smile-model (OLS/fit (Formula/lhs "y") data-frame))

(def gpm (smile/model->gpm smile-model "y" (.names data-frame)))
```

Finally, launch the InferenceQL REPL:

``` clojure
(main/repl data {:model gpm})
```

``` sql
select * from generate y under model conditioned by x1=1 and x2=1 and x3=1 and x4=1 and x5=1 and x6=1 and x7=1 and x8=1 and x9=1 and x10=1 limit 5;
```
    
