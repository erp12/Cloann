(ns cloann.nn-reporting
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.operators :as mat-ops]
            [cloann.util :as util])
  (:use [incanter core stats charts]))


(defn plot-nn-evaluations
  "Saves line plots of all 6 types of errors over each epoch."
  [training-error training-classification-error testing-error testing-classification-error validation-error validation-classification-error]
  (let [dateString (str (util/get-date-time-string))
        dir (str "charts/" dateString)]
    (do
      ; Make folder for this run
      (.mkdir (java.io.File. (str "charts/" dateString)))
      ; Save plot of training error.
      (save (line-chart (range (count training-error))
                        training-error
                        :title "Training Error"
                        :x-label "Epoch"
                        :y-label "Error")
            (str dir "/training_error.png")
            :width 800
            :height 600)
      ; Save plot of training classification error.
      (save (line-chart (range (count training-classification-error))
                        training-classification-error
                        :title "Training Classification Error"
                        :x-label "Epoch"
                        :y-label "Classification Error")
            (str dir "/training_classification_error.png")
            :width 800
            :height 600)
      ; Save plot of testing error.
      (save (line-chart (range (count testing-error))
                        testing-error
                        :title "Testing Error"
                        :x-label "Epoch"
                        :y-label "Error")
            (str dir "/testing_error.png")
            :width 800
            :height 600)
      ; Save plot of testing classification error.
      (save (line-chart (range (count testing-classification-error))
                        testing-classification-error
                        :title "Testing Classification Error"
                        :x-label "Epoch"
                        :y-label "Classification Error")
            (str dir "/testing_classification_error.png")
            :width 800
            :height 600)
      ; Save plot of validation error.
      (save (line-chart (range (count validation-error))
                        validation-error
                        :title "Validation Error"
                        :x-label "Epoch"
                        :y-label "Error")
            (str dir "/validation_error.png")
            :width 800
            :height 600)
      ; Save plot of validation classification error.
      (save (line-chart (range (count validation-classification-error))
                        validation-classification-error
                        :title "Validation Classification Error"
                        :x-label "Epoch"
                        :y-label "Classification Error")
            (str dir "/validation_classification_error.png")
            :width 800
            :height 600))))