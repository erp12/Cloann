(ns cloann.examples.iris
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

;; Matrix of all the data from the csv file
(def training-data-matrix
  (dIO/csv->matrix "data/iris_training.csv" false))
(def testing-data-matrix
  (dIO/csv->matrix "data/iris_testing.csv" false))
(def validation-data-matrix
  (dIO/csv->matrix "data/iris_validation.csv" false))

;(dIO/create-data-sub-set-from-full-matrix training-data-matrix
;                                          [0 1 2 3]
;                                          [4 5 6]
;                                          (vec (range (count [4 5 6])))
;                                          (repeat 1)))

(def nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices training-data-matrix
                                                    testing-data-matrix
                                                    validation-data-matrix
                                                    [0 1 2 3] ; Input indexes
                                                    [4 5 6]) ; Output indexes
   :max-epochs 1000
   :max-weight-intial 0.5
   :learning-rate 0.01})

;(cloann/run-cloann nn-params)