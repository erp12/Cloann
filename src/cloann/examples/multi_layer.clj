(ns cloann.examples.multi_layer_iris
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]
            [cloann.transfer-functions :as tran-funcs]))

;; Matrix of all the data from the csv file
(def training-data-matrix
  (dIO/csv->matrix "data/iris_training.csv" false))
(def testing-data-matrix
  (dIO/csv->matrix "data/iris_testing.csv" false))
(def validation-data-matrix
  (dIO/csv->matrix "data/iris_validation.csv" false))

(def topology-encoding 
  {:layers  {:I {:num-nodes 4}
             :H1 {:num-nodes 5}
             :O {:num-nodes 3}}
   :layer-connections [[:I  :H1]
                       [:H1 :O]]})
(def nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices training-data-matrix
                                                    testing-data-matrix
                                                    validation-data-matrix
                                                    [0 1 2 3] ; Input indexes
                                                    [4 5 6]) ; Output indexes
   :topology-encoding topology-encoding
   :max-epochs 500
   :max-weight-intial 0.5
   :learning-rate 0.02
   :validation-stop-threshold 0.03})

(swap! cloann/nn-params #(merge % nn-params))
;(cloann/run-cloann nn-params)

(def uwm (cloann/generate-uninitialized-weight-matrix (:layers (:topology-encoding @cloann/nn-params))
                                                      (:layer-connections (:topology-encoding @cloann/nn-params))))

;(util/matrix-2d-pretty-print uwm)

(def wm (cloann/initialize-weights uwm
                                   (:max-weight-intial @cloann/nn-params)))

(def ff (cloann/feed-forward (first (:inputs (:training-set (:data-sets @cloann/nn-params))))
                             wm))

ff
