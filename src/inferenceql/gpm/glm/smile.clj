(ns inferenceql.gpm.glm.smile
  "Functions for creating GPMs from SMILE models."
  (:import [smile.data DataFrame]
           [smile.glm GLM]
           [smile.regression LinearModel])
  (:require [inferenceql.gpm.glm :as glm]))

(defprotocol ToGPM
  :extend-via-metadata true
  (model->gpm [model dependent-variable variables]
    "Creates a GPM from a SMILE model."))

(extend LinearModel
  ToGPM
  {:model->gpm
   (fn [model dependent-variable variables]
     (let [dependent-variable (keyword dependent-variable)
           variables (map keyword variables)

           coefficients (.coefficients model)
           independent-variable->coeffcient (dissoc (zipmap variables coefficients)
                                                    dependent-variable)
           bias-term (.intercept model)
           sigma (.error model)]
       (glm/linear-regression
        {:dependent-variable dependent-variable
         :independent-variable->coefficient independent-variable->coeffcient
         :bias-term bias-term
         :sigma sigma})))})

(extend GLM
  ToGPM
  {:model->gpm
   (fn [model dependent-variable variables]
     (let [dependent-variable (keyword dependent-variable)
           variables (map keyword variables)

           coefficients (.coefficients model)
           independent-variable->coeffcient (dissoc (zipmap variables coefficients)
                                                    dependent-variable)
           bias-term (get-in (.ztest model) [0 0])]
       (glm/logistic-regression
        {:dependent-variable dependent-variable
         :independent-variable->coefficient independent-variable->coeffcient
         :bias-term bias-term})))})

(defn data-frame?
  "Returns `true` if `x` is an instance of `smile.data.DataFrame`."
  [x]
  (instance? smile.data.DataFrame x))

(defn vectors->data-frame
  "Produces a `smile.data.DataFrame` from the provided two-dimensional
  sequence. All values in `seqs` must be compatible with the
  component type. Class objects for the primitive types can be
  obtained using, e.g., `Integer/TYPE`."
  [type names seqs]
  (let [data (->> seqs
                  (map #(into-array type %))
                  (into-array))]
    (DataFrame/of data (into-array java.lang.String (map name names)))))

(defn data-frame->vectors
  "Returns the rows in a `smile.data.DataFrame` as a sequence of
  vectors."
  [data-frame]
  (for [i (range (.nrows data-frame))]
    (for [j (range (.ncols data-frame))]
      (.get data-frame i j))))

(defn data-frame->maps
  "Returns the rows in a `smile.data.DataFrame` as a sequence of
  maps."
  [data-frame]
  (map #(zipmap (.names data-frame) %)
       (data-frame->vectors data-frame)))
