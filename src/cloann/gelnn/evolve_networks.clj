(ns cloann.gelnn.evolve-networks
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
(def training-data-matrix
  (dIO/csv->matrix "data/iris_training.csv" false))
(def testing-data-matrix
  (dIO/csv->matrix "data/iris_testing.csv" false))
(def validation-data-matrix
  (dIO/csv->matrix "data/iris_validation.csv" false))

; EMBRYO ENCODING
(def num-inputs 4)
(def num-outputs 3)
(def embryo-encoding
  {:layers  {:I {:num-nodes num-inputs}
             :O {:num-nodes num-outputs}}
   :layer-connections [[:I  :O]]})
(def data-sets
  (dIO/create-data-sets-from-3-matrices training-data-matrix
                                        testing-data-matrix
                                        validation-data-matrix
                                        [0 1 2 3] ; Input indexes
                                        [4 5 6])) ; Output indexes

; CLOJUSH ATOM GENERATORS
(def gelnn-atom-generators
  (concat (list 
            'nn_connect_layers
            'nn_bud
            'nn_split
            'nn_loop
            'nn_reverse
            'nn_set_num_nodes_layer
            (fn [] (lrand-int 100)))))
          ;(registered-for-stacks [:integer :exec])))

; CLOJUSH ERROR FUNCTIONS
(defn network-evo-error-function
  [program]
  (let [final-state (run-push program 
                              (push-item embryo-encoding
                                 :auxilary
                                 (make-push-state)))
        topology-encoding (first (:auxilary final-state))
        nn-params {:data-sets data-sets
                   :topology-encoding topology-encoding
                   :max-epochs 200
                   :max-weight-initial 0.1
                   :learning-rate 0.1
                   :validation-stop-threshold 0.01}
        training-result (cloann/run-cloann nn-params false)]
    (if (:solution-found training-result)
      (do
        (println "||NAILED IT||")
        [0 0 0])
      [(:final-validation-error training-result)
       (:final-testing-error training-result)
       (:final-training-error training-result)])))

(defn gelnn-report
  "Custom generational report."
  [best population generation error-function report-simplifications]
  (let [best-program (not-lazy (:program best))
        best-test-errors (error-function best-program :test)
        best-total-test-error (apply +' best-test-errors)]
    (println ";;******************************")
    (printf ";; -*- GELNN Iris Report - generation %s\n" generation)(flush)
    (println "Test total error for best:" best-total-test-error)
    (println (format "Test mean error for best: %.5f" (double (/ best-total-test-error (count best-test-errors)))))
    (when (zero? (:total-error best))
      (doseq [[i error] (map vector
                             (range)
                             best-test-errors)]
        (println (format "Test Case  %3d | Error: %s" i (str error)))))
    (println ";;------------------------------")
    (println "Outputs of best individual on training cases:")
    (error-function best-program :train true)
    (println ";;******************************")
    ))

; 
(def argmap
  {:use-single-thread true
   :error-function network-evo-error-function
   :atom-generators gelnn-atom-generators
   :max-points 200
   :max-genome-size-in-initial-program 50
   :evalpush-limit 200
   :population-size 5
   :max-generations 100
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
   :final-report-simplifications 5000})

(pushgp argmap)
