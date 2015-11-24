(ns cloann.nn-reporting
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.operators :as mat-ops]
            [cloann.util :as util])
  (:use [incanter core charts]))

(defn plot-nn-evaluations
  "Saves line plots of all 6 types of errors over each epoch."
  [training-error training-classification-error testing-error testing-classification-error validation-error validation-classification-error]
  (let [dateString (str (util/get-date-time-string))
        dir (str "nn_reporting/")
        x-vals (vec (range (count training-error)))]
    (.mkdir (java.io.File. "nn_reporting/"))
    (-> 
      (xy-plot x-vals
               training-error
               :title "Errors"
               :x-label "Epoch"
               :y-label "Error"
               :legend true
               :series-label "Training Error")
      (add-lines x-vals
                  training-classification-error
                  :series-label "Training Classification Error")
      (add-lines x-vals
                  testing-error
                  :series-label "Testing Error")
      (add-lines x-vals
                  testing-classification-error
                  :series-label "Testing Classification Error")
      (add-lines x-vals
                  validation-error
                  :series-label "Validation Error")
      (add-lines x-vals
                  validation-classification-error
                  :series-label "Validation Classification Error")
      (save (str dir dateString ".png")
            :width 1280
            :height 720))))
