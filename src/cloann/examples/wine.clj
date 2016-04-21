(ns cloann.examples.wine
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]))

;; Matrix of all the data from the csv file
(def data-matrix
  (dIO/csv->matrix "data/wine_norm.csv" false))

(def topology-encoding 
  {:layers  {:I {:num-nodes 12}
             :H1 {:num-nodes 1}
             :O {:num-nodes 3}}
   :layer-connections [[:I  :H1]
                       [:H1 :O]]})

(def nn-params
  {:data-sets (dIO/create-data-sets-from-1-matrix data-matrix
                                                  [3 4 5 6 7 8 9 10 11 12 13 14]; Input indexes
                                                  [0 1 2] ; Output indexes
                                                  100     ; Number of observations to take for training
                                                  100     ; Number of observations to take for testing
                                                  100)    ; Number of observations to take for validation
   :topology-encoding topology-encoding
   :max-epochs 3
   :max-weight-intial 1
   :learning-rate 0.1
   :validation-stop-threshold 0.03})

;(cloann/run-cloann nn-params true)