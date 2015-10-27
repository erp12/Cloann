(ns cloann.examples.xor
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

;; Matrix of all the data from the csv file
(def data-matrix
  (dIO/csv->matrix "XOR_no_heading.csv" true))

(def nn-params
  (reset!
    cloann/nn-params
    (merge @cloann/nn-params
         {:data-set (dIO/create-data-set-from-matrix data-matrix
                                                     [0 1] ; Input indexes
                                                     [2]   ; Output indexes
                                                     500   ; Number of observations to take for training
                                                     100   ; Number of observations to take for testing
                                                     100)}))); Number of observations to take for validation