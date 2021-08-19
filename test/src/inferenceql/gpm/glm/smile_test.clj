(ns inferenceql.gpm.glm.smile-test
  (:import [smile.data.formula Formula]
           [smile.glm GLM]
           [smile.glm.model Bernoulli]
           [smile.io Read]
           [smile.regression OLS])
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is] :as test]
            [inferenceql.gpm.glm.smile :as smile]
            [inferenceql.inference.gpm :as gpm]))

(deftest vectors->data-frame
  (is (smile/data-frame?
       (smile/vectors->data-frame
        Integer/TYPE
        [:x :y :z]
        [[1 0 2]
         [0 1 0]
         [2 0 1]]))))

(deftest round-trip
  (let [vs0 [[1 3 2]
             [4 1 5]
             [2 6 1]]
        vs1 (->> vs0
                 (smile/vectors->data-frame Integer/TYPE [:x :y :z])
                 (smile/data-frame->vectors))]
    (is (= vs0 vs1))))

(deftest data-frame->maps
  (let [vs [[1 0 2]
            [0 1 0]
            [2 0 1]]
        ms [{"x" 1 "y" 0 "z" 2}
            {"x" 0 "y" 1 "z" 0}
            {"x" 2 "y" 0 "z" 1}]]
    (is (= ms (smile/data-frame->maps
               (smile/vectors->data-frame
                Integer/TYPE
                ["x" "y" "z"]
                vs))))))

(deftest linear-model
  (let [data-frame (Read/arff (str (io/resource "2dplanes.arff")))
        variables (set (.names data-frame))
        independent-variables (disj variables "y")
        model (OLS/fit (Formula/lhs "y") data-frame)
        gpm (smile/model->gpm model "y" (.names data-frame))
        conditions (zipmap (map keyword independent-variables) (repeat 0))]
    (is (number? (gpm/logpdf gpm {:y 0} conditions)))
    (is (number? (:y (gpm/simulate gpm [:y] conditions))))
    (is (number? (:y (gpm/simulate (gpm/condition gpm conditions) [:y] {}))))))

(deftest generalized-linear-model
  (let [data-frame (let [names ["x" "y" "z"]
                         generate-cell #(rand-int 2)
                         generate-row #(repeatedly (count names) generate-cell)
                         table (repeatedly 20 generate-row)]
                     (smile/vectors->data-frame Integer/TYPE names table))
        variables (set (.names data-frame))
        independent-variables (disj variables "x")
        model (GLM/fit (Formula/lhs "x") data-frame (Bernoulli/logit))
        gpm (smile/model->gpm model "x" (.names data-frame))
        conditions (zipmap (map keyword independent-variables) (repeat 0))]
    (is (number? (gpm/logpdf gpm {:x 0} conditions)))
    (is (boolean? (:x (gpm/simulate gpm [:x] conditions))))
    (is (boolean? (:x (gpm/simulate (gpm/condition gpm conditions) [:x] {}))))))
