(ns inferenceql.gpm.glm
  "Linear model GPMs, both generalized and otherwise."
  (:require [clojure.set :as set]
            [inferenceql.gpm.glm.math :as math]
            [inferenceql.inference.gpm.conditioned :as conditioned]
            [inferenceql.inference.gpm.constrained :as constrained]
            [inferenceql.inference.gpm.proto :as proto]
            [inferenceql.inference.primitives :as primitives]))

(defn ^:private linear-predictor
  [{:keys [independent-variable->coefficient bias-term]} conditions]
  (+ bias-term
     (->> independent-variable->coefficient
          (map (fn [[variable coefficient]]
                 (* coefficient
                    (get conditions variable))))
          (reduce +))))

(defrecord GLM [dependent-variable independent-variable->coefficient logpdf simulate bias-term link-function]
  proto/Variables
  (variables [_]
    (set (conj (keys independent-variable->coefficient)
               dependent-variable)))

  proto/GPM
  (logpdf [{:keys [logpdf] :as this} targets conditions]
    (when-not (= [dependent-variable] (keys targets))
      (throw (ex-info "LogPDF only supported for the dependent variable."
                      {:dependent-variable dependent-variable
                       :targets targets})))

    (let [independent-variables (set (keys independent-variable->coefficient))]
      (when-not (set/subset? independent-variables (set (keys conditions)))
        (throw (ex-info "LogPDF only supported when all independent variables are conditioned."
                        {:independent-variables independent-variables
                         :conditions conditions}))))

    (let [mean (-> (linear-predictor this conditions)
                   (link-function))
          value (get targets dependent-variable)]
      (logpdf value mean)))

  (simulate [{:keys [simulate] :as this} targets conditions]
    (when-not (= [dependent-variable] (seq targets))
      (throw (ex-info "Simulation of independent variables is not yet supported."
                      {:dependent-variable dependent-variable
                       :targets targets
                       :cognitect.anomalies/category :cognitect.anomalies/unsupported})))

    (let [independent-variables (set (keys independent-variable->coefficient))]
      (when-not (set/subset? independent-variables (set (keys conditions)))
        (throw (ex-info "Simulation when independent variables are unconditioned is not yet supported."
                        {:independent-variables independent-variables
                         :conditions conditions
                         :cognitect.anomalies/category :cognitect.anomalies/unsupported}))))

    (let [mean (-> (linear-predictor this conditions)
                   (link-function))
          value (simulate mean)]
      {dependent-variable value}))

  proto/Condition
  (condition [this conditions]
    (conditioned/condition this conditions))

  proto/Constrain
  (constrain [this targets conditions]
    (constrained/constrain this targets conditions)))

(defn glm?
  "Returns `true` if `x` is a GLM GPM. Otherwise returns `false`."
  [x]
  (instance? GLM x))

(defn linear-regression
  [{:keys [dependent-variable independent-variable->coefficient bias-term sigma]}]
  (-> (map->GLM {:dependent-variable dependent-variable
                 :independent-variable->coefficient independent-variable->coefficient
                 :bias-term bias-term

                 :link-function identity
                 :logpdf (fn [x mean] (primitives/gaussian-logpdf x {:mu mean :sigma sigma}))
                 :simulate (fn [mean] (primitives/gaussian-simulate {:mu mean :sigma sigma}))})
      (assoc ::type ::linear-regression)))

(defn linear-regression?
  "Returns true if `x` is a linear regression GPM."
  [x]
  (and (glm? x) (= ::linear-regression (::type x))))

(defn logistic-regression
  [{:keys [dependent-variable independent-variable->coefficient bias-term]}]
  (-> (map->GLM {:dependent-variable dependent-variable
                 :independent-variable->coefficient independent-variable->coefficient
                 :bias-term bias-term

                 :link-function math/sigmoid
                 :logpdf (fn [x p] (primitives/bernoulli-logpdf x {:p p}))
                 :simulate (fn [p] (primitives/bernoulli-simulate {:p p}))})
      (assoc ::type ::logistic-regression)))

(defn logistic-regression?
  "Returns true if `x` is a logistic regression GPM."
  [x]
  (and (glm? x) (= ::logistic-regression (::type x))))
