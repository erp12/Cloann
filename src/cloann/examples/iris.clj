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

(def nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices training-data-matrix
                                                    testing-data-matrix
                                                    validation-data-matrix
                                                    [0 1 2 3] ; Input indexes
                                                    [4 5 6]) ; Output indexes
   :max-epochs 500
   :max-weight-intial 0.5
   :learning-rate 0.02
   :validation-stop-threshold 0.03})

(cloann/run-cloann nn-params)

;(util/data-set-pretty-print (:testing-set (:data-sets nn-params)))