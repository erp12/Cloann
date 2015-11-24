(ns cloann.examples.wine
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

;; Matrix of all the data from the csv file
(def data-matrix
  (dIO/csv->matrix "data/wine.csv" false))

(def nn-params
  {:data-sets (dIO/create-data-sets-from-1-matrix data-matrix
                                                  (vec (range 12)) ; Input indexes
                                                  [12] ; Output indexes
                                                  200   ; Number of observations to take for training
                                                  100   ; Number of observations to take for testing
                                                  100)  ; Number of observations to take for validation
   :max-epochs 500
   :max-weight-intial 1
   :learning-rate 0.1
   :validation-stop-threshold 0.1
   :num-hidden-nodes 4})

(cloann/run-cloann nn-params)

;(util/data-set-pretty-print (:testing-set (:data-sets nn-params)))