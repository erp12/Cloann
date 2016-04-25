(ns cloann.examples.xor
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]))

;; Matrix of all the data from the csv file
(def data-matrix
  [[0 0 0]
   [0 1 1]
   [1 0 1]
   [1 1 0]])

(def topology-encoding 
  {:layers  {:I {:num-nodes 2}
             :H1 {:num-nodes 2}
             :O {:num-nodes 1}}
   :layer-connections [[:I  :H1]
                       [:H1 :O]]})

(def nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices data-matrix
                                                    data-matrix
                                                    data-matrix
                                                    [0 1] ; Input indexes
                                                    [2]) ; Output indexes
   :max-epochs 1000
   :max-weight-initial 0.1
   :learning-rate 0.5
   :validation-stop-threshold 0.08})

(cloann/run-cloann nn-params topology-encoding  true)