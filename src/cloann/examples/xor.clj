(ns cloann.examples.xor
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]))

;; Matrix of all the data from the csv file
(def data-matrix
  (dIO/csv->matrix "data/XOR_with_headings.csv" true))

(def nn-params
  {:data-sets (dIO/create-data-sets-from-1-matrix data-matrix
                                                  [0 1] ; Input indexes
                                                  [2 3]   ; Output indexes
                                                  500   ; Number of observations to take for training
                                                  300   ; Number of observations to take for testing
                                                  300)}); Number of observations to take for validation

;(cloann/run-cloann nn-params)