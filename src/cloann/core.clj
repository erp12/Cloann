(ns cloann.core
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.nn-reporting :as report])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

; Default params to the neural network.
(def nn-params
  (atom 
    {:activation-func act-funcs/hyperbolic-tangent
     :activation-func-derivative act-funcs/hyperbolic-tangent-derivative
     :max-weight 1
     :data-set nil
     :sample-limit 10
     :plot-graphs true
     :learning-rate 0.1}))

(defn feed-forward
  "Computes the output of the neural network given the inputs and the weights."
  [input-matrix weight-matrix bias-node-matrix]
  (let [net (* weight-matrix
               (util/horizontal-matrix-concatenation input-matrix bias-node-matrix))
        output (emap (:activation-func @nn-params) net)]
    [net output]))

(defn generate-initial-weight-matrix
  "Generates matrix of weights between max-weight and negative max-weight"
  [max-weight matrix-width matrix-height]
  (matrix
    (vec
      (repeatedly matrix-height
                  (fn []
                    (vec
                      (repeatedly matrix-width 
                                  (fn []
                                    (- (rand (* max-weight 2))
                                       max-weight)))))))))

(defn evaluate-network
  "Returns the error, and classification error of the network on a particular data-set"
  [data-set weight-matrix]
  (let [; Store the result of feeding the data-set into the network
        ff-result (feed-forward (:inputs data-set)
                                weight-matrix)
        ; Pull out the output from the feed forward result
        output (second ff-result)
        ; Pull out the resulting network from the feed forward.
        net (first ff-result)
        ; Calcuate the error of the network on the data set.
        error (/ (util/sum-all-2D-matrix-components (Math/pow (- (:outputs data-set)
                                                                 output)
                                                              2))
                 (* (count (:outputs data-set)) (:output-count data-set)))
        ; Find the classes of the outputs.
        classes (util/output->class output)
        ; User outputs to find the classification error on the data set.
        classification-error (/ (count (filter true? (map = classes (:classes data-set))))
                                (count (:outputs data-set)))]
    [error classification-error]))

(defn backpropagation 
  "I don't do anything right now. Come back later."
  [input-matrix weight-matrix learning-rate bias-vector]
  weight-matrix)

(defn train-nn
  [data-sets]
  (let [; Inital randomized weights to the network
        init-weight-matrix (generate-initial-weight-matrix 1
                                                      (inc (:input-counts data-sets))
                                                      (:output-count data-sets))
        ; Bias vectors for the various data-sets
        training-bias (repeat (:count (:training data-sets)) 1)
        testing-bias (repeat (:count (:testing data-sets)) 1)
        validation-bias (repeat (:count (:validation data-sets)) 1)]
    (loop [; nn's training epoch
           epoch 0
           ; unputs to the network
           inputs (:inputs (:training data-sets))
           ; current set of weights for the network
           weights init-weight-matrix
           ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
           ; Tracking info for reporting later
           training-error []
           training-classification-error []
           testing-error []
           testing-classification-error []
           validation-error []
           validation-classification-error []]
      (if (= epoch 500)
        (do
          (println "Training finished. Now draw some graphs.")
          (report/plot-nn-evaluations training-error
                                      training-classification-error
                                      testing-error
                                      testing-classification-error
                                      validation-error
                                      validation-classification-error))
        (let [training-eval (evaluate-network (:training data-sets) weights)
              testing-eval (evaluate-network (:testing data-sets) weights)
              validation-eval (evaluate-network (:validation data-sets) weights)]
          (recur 
          ; Increment the epoch number
          (inc epoch)
          ; Inputs stay the same
          inputs
          ; Apply the results of the backpropagation as the new weights
          (backpropagation inputs 
                           weights 
                           (:learning-rate @nn-params))
          ;;;;;;;;;;;;;;;;;;;;;;;;
          ; Append errors to vectors for reporting
          (conj training-error (first training-eval))
          (conj training-classification-error (second training-eval))
          (conj testing-error (first testing-eval))
          (conj testing-classification-error (second testing-eval))
          (conj validation-error (first validation-eval))
          (conj validation-classification-error (second validation-eval))))))))

(defn run-cloann
  [params]
  (swap! nn-params #(merge % params))
  (train-nn (:data-sets @nn-params)))


(defn -main 
  []
  (println "I don't do anything yet..."))