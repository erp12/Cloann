(ns cloann.examples.evolve-networks-student
  (:use [clojush.ns]
        [clojush translate]
        cloann.gelnn.nn-instructions)
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]))

(use-clojush)


; DATA SETS
(def data-matrix
  [[0 0 0]
   [0 1 1]
   [1 0 1]
   [1 1 0]])


; EMBRYO ENCODING
(def num-inputs 2)
(def num-outputs 1)
(def embryo-encoding
  {:layers  {:I {:num-nodes num-inputs}
             :O {:num-nodes num-outputs}}
   :layer-connections [[:I  :O]]})
(def data-sets
  (dIO/create-data-sets-from-3-matrices data-matrix
                                        data-matrix
                                        data-matrix
                                        [0 1] ; Input indexes
                                        [2])) ; Output indexes

; CLOJUSH ATOM GENERATORS
(def gelnn-atom-generators
  (concat (list 
            'nn_connect_layers
            'nn_bud
            'nn_split
            'nn_loop
            'nn_reverse
            'nn_set_num_nodes_layer
            (fn [] (lrand-int 100))
            (fn [] (lrand-int 10)))
          (registered-for-stacks [:integer :exec])))

; CLOJUSH ERROR FUNCTIONS
(defn student-network-evo-error-function
  [program]
  (let [final-state (run-push program 
                              (push-item embryo-encoding
                                 :auxilary
                                 (make-push-state)))
        topology-encoding (first (:auxilary final-state))
        nn-params {:data-sets data-sets
                   :max-epochs 1000
                   :max-weight-initial 0.1
                   :learning-rate 0.5
                   :validation-stop-threshold 0.08}
        training-result (cloann/run-cloann nn-params topology-encoding false)]
    (if (:solution-found training-result)
      (do
        (println "||NAILED IT||")
        [0 0 0])
      [(:final-validation-error training-result)
       (:final-testing-error training-result)
       (:final-training-error training-result)])))

(def argmap
  {:use-single-thread false 
   :error-function student-network-evo-error-function
   :atom-generators gelnn-atom-generators
   :max-points 200
   :max-genome-size-in-initial-program 80
   :evalpush-limit 200
   :population-size 50
   :max-generations 50
   :parent-selection :lexicase
   :genetic-operator-probabilities {:alternation 0.2
                                    :uniform-mutation 0.2
                                    :uniform-close-mutation 0.1
                                    [:alternation :uniform-mutation] 0.5}
   :alternation-rate 0.01
   :alignment-deviation 10
   :uniform-mutation-rate 0.01
   :print-behavioral-diversity false
   :report-simplifications 0
   :final-report-simplifications 40
   :return-simplified-on-failure true})

(pushgp argmap)
