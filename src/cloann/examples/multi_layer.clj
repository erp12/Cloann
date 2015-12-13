(ns cloann.examples.multi_layer_iris
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]))

;; Matrix of all the data from the csv file
(def training-data-matrix
  (dIO/csv->matrix "data/iris_training.csv" false))
(def testing-data-matrix
  (dIO/csv->matrix "data/iris_testing.csv" false))
(def validation-data-matrix
  (dIO/csv->matrix "data/iris_validation.csv" false))

(def layers  {:I {:num-inputs 4
                  :num-outputs 5
                  ;:first-row 0
                  ;:first-col 0
                  :activation-func act-funcs/hyperbolic-tangent
                  :activation-func-derivative act-funcs/hyperbolic-tangent-derivative}
              :H1 {:num-inputs 5
                   :num-outputs 4
                   ;:first-row 4
                   ;:first-col 5
                   :activation-func act-funcs/hyperbolic-tangent
                   :activation-func-derivative act-funcs/hyperbolic-tangent-derivative}
              :O {:num-inputs 4
                   :num-outputs 3
                   ;:first-row 9
                   ;:first-col 9
                   :activation-func act-funcs/hyperbolic-tangent
                   :activation-func-derivative act-funcs/hyperbolic-tangent-derivative}})
  
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