(ns inferenceql.gpm.glm.math
  (:import [org.apache.commons.math3.analysis.function Sigmoid]))

(defn sigmoid
  [x]
  (.value (Sigmoid. 0 1)
          (double x)))
