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
  [inputs weight-matrix bias-node-matrix]

  (let [temp (util/horizontal-matrix-concatenation inputs bias-node-matrix)
        net (inner-product temp 
                           weight-matrix)
        output (emap (:activation-func @nn-params) net)]
    [net output]))

(defn generate-initial-weight-matrix
  "Generates matrix of weights between max-weight and negative max-weight"
  [max-weight num-rows num-cols]
  (array
    (vec
      (repeatedly num-rows
                  (fn []
                    (vec
                      (repeatedly num-cols
                                  (fn []
                                    (- (rand (* max-weight 2))
                                       max-weight)))))))))

(defn evaluate-network
  "Returns the error, and classification error of the network on a particular data-set"
  [data-set weight-matrix bias]
  (let [; Store the result of feeding the data-set into the network
        ff-result (feed-forward (array (:inputs data-set))
                                weight-matrix
                                bias)
        ; Pull out the outputs from the feed forward result
        outputs (second ff-result)
        ; Pull out the resulting network from the feed forward.
        net (first ff-result)
        ; Calcuate the error of the network on the data set.
        error (/ (util/sum-all-2D-matrix-components (emap square
                                                            (- (:outputs data-set)
                                                               outputs)))
                   (* (count (:outputs data-set))
                      (:output-count (:data-sets @nn-params))))
        ; Find the classes of the outputs.
        classes (map util/output->class (transpose outputs))
        ; Uses outputs to find the classification error on the data set.
        classification-error (do
                               ;(println "Classes:" classes)
                               ;(println "Target Classes:" (:classes data-set))
                               (/ (count (filter true? (map = classes (:classes data-set))))
                                  (count (:outputs data-set))))]
    [error classification-error]))

(defn backpropagation 
  "I don't do anything right now. Come back later."
  [inputs-matrix outputs-matrix weight-matrix learning-rate bias-vector]
  (let [; Pick random sample 
        rand-sample-index (rand-int (:input-count (:data-sets @nn-params)))
        ; Feed sample through network
        ff-result (feed-forward [(nth inputs-matrix rand-sample-index)]
                                weight-matrix
                                bias-vector)
        output (second ff-result)
        net (first ff-result)
        ; Take vector of how far off feed foward was
        error-vector (- (transpose (nth outputs-matrix rand-sample-index))
                        output)
        ; How should the weight change
        delta (emap *
                    error-vector
                    (emap (:activation-func-derivative @nn-params) net))
        ; Put in temp for code readability
        temp (transpose (conj (nth inputs-matrix rand-sample-index)
                               (nth bias-vector rand-sample-index)))
        ; How much should the weights change
        weights-delta (* learning-rate
                         (outer-product temp
                                        delta))
        ; What should the new weights be
        new-weights (emap + weight-matrix 
                          (reshape weights-delta 
                                   (shape weight-matrix)))]
    new-weights))

(defn train-nn
  [data-sets]
  (let [; Inital randomized weights to the network
        init-weight-matrix (generate-initial-weight-matrix 1
                                                           (inc (:input-count data-sets))
                                                           (:output-count data-sets))
        ; Bias vectors for the various data-sets
        training-bias (vec (repeat (:count (:training-set data-sets)) 1))
        testing-bias (vec (repeat (:count (:testing-set data-sets)) 1))
        validation-bias (vec (repeat (:count (:validation-set data-sets)) 1))]
    (loop [; nn's training epoch
           epoch 0
           ; inputs to the network
           inputs (:inputs (:training-set data-sets))
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
        (let [training-eval (evaluate-network (:training-set data-sets) weights training-bias)
              testing-eval (evaluate-network (:testing-set data-sets) weights testing-bias)
              validation-eval (evaluate-network (:validation-set data-sets) weights validation-bias)]
          (println "Epoch:" epoch)
          (recur 
            ; Increment the epoch number
            (inc epoch)
            ; Inputs stay the same
            inputs
            ; Apply the results of the backpropagation as the new weights
            (backpropagation inputs
                             (:outputs (:training-set data-sets))
                             weights 
                             (:learning-rate @nn-params)
                             training-bias)
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
  ;(println (:data-sets @nn-params))
  (train-nn (:data-sets @nn-params)))


(defn -main 
  []
  (println "I don't do anything yet..."))