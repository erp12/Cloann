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
(def training-data-matrix
  (dIO/csv->matrix "data/norm_student_alc_training.csv" true))
(def testing-data-matrix
  (dIO/csv->matrix "data/norm_student_alc_testing.csv" true))
(def validation-data-matrix
  (dIO/csv->matrix "data/norm_student_alc_validation.csv" true))

; EMBRYO ENCODING
(def num-inputs 24)
(def num-outputs 3)
(def embryo-encoding
  {:layers  {:I {:num-nodes num-inputs}
             :O {:num-nodes num-outputs}}
   :layer-connections [[:I  :O]]})
(def data-sets
  (dIO/create-data-sets-from-3-matrices training-data-matrix
                                        testing-data-matrix
                                        validation-data-matrix
                                        (vec (range 24)) ; Input indexes
                                        [24 25 26])) ; Output indexes

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
(defn student-network-evo-error-function
  [program]
  (let [final-state (run-push program 
                              (push-item embryo-encoding
                                 :auxilary
                                 (make-push-state)))
        topology-encoding (first (:auxilary final-state))
        nn-params {:data-sets data-sets
                   :topology-encoding topology-encoding
                   :max-epochs 500
                   :max-weight-initial 0.15
                   :learning-rate 0.01
                   :validation-stop-threshold 0.002}
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
  {:use-single-thread false 
   :error-function student-network-evo-error-function
   :atom-generators gelnn-atom-generators
   :max-points 200
   :max-genome-size-in-initial-program 40
   :evalpush-limit 200
   :population-size 100
   :max-generations 500
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
   :final-report-simplifications 500
   :return-simplified-on-failure true})

(pushgp argmap)
