(ns inferenceql.gpm.glm-test
  (:require [clojure.test :refer [deftest is testing] :as test]
            [inferenceql.gpm.glm :as glm]
            [inferenceql.inference.gpm :as gpm]
            [inferenceql.inference.gpm.primitive-gpms.gaussian :as gaussian]))

(deftest linear-regression?
  (let [glm (glm/linear-regression
             {:dependent-variable :x
              :bias-term 1

              :independent-variables
              {:coefficient 3
               :model (gaussian/spec->gaussian :y :suff-stats {:n 0 :sum-x 0 :sum-x-sq 0})}})]
    (is (glm/linear-regression? glm))))

(deftest logistic-regression?
  (let [glm (glm/logistic-regression
             {:dependent-variable :x
              :independent-variable->coefficient {:y 3 :z 2}
              :bias-term 7})]
    (is (glm/logistic-regression? glm))))

(deftest logpdf
  (let [glm (glm/linear-regression
             {:dependent-variable :x
              :independent-variable->coefficient {:y 3 :z 2}
              :bias-term 7})]
    (testing "incomplete constraints throws exception"
      (is (thrown? Exception (gpm/logpdf glm {:x 0} {:y 0}))))

    (testing "logpdf on non-dependent variable throws exception"
      (is (thrown? Exception (gpm/logpdf glm {:y 0} {:z 0}))))))
