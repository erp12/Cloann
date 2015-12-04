(ns cloann.examples.wine
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]))

;; Matrix of all the data from the csv file
(def data-matrix
  (dIO/csv->matrix "data/wine_norm.csv" false))

(def nn-params
  {:data-sets (dIO/create-data-sets-from-1-matrix data-matrix
                                                  [3 4 5 6 7 8 9 10 11 12 13 14]; Input indexes
                                                  [0 1 2] ; Output indexes
                                                  100     ; Number of observations to take for training
                                                  100     ; Number of observations to take for testing
                                                  100)    ; Number of observations to take for validation
   :max-epochs 1000
   :max-weight-intial 1
   :learning-rate 0.02
   :validation-stop-threshold 0.03
   :num-hidden-nodes 1})

;(cloann/run-cloann nn-params)

;(util/data-set-pretty-print (:testing-set (:data-sets nn-params)))