(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(def W (matrix [[0 1 2]
                [3 5 4]
                [8 7 6]]))


;; Matrix of all the data from the csv file
(def data-matrix
  (dIO/csv->matrix "XOR_no_heading.csv" true))

(def ds (dIO/create-data-sets-from-matrix data-matrix
                                         [0 1] ; Input indexes
                                         [2]   ; Output indexes
                                         20   ; Number of observations to take for training
                                         10   ; Number of observations to take for testing
                                         10)) ; Number of observations to take for validation

(:count (:training ds))