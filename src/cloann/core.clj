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
     :max-weight-initial 0.5
     :data-set nil
     :sample-limit 10
     :plot-graphs true
     :learning-rate 0.01
     :max-epochs 1000
     :debug-prints true
     :target-error-threshold 0.05}))

(defn feed-forward
  "Computes the output of the neural network given the inputs and the weights."
  [inputs weight-matrix bias-node-matrix]
  (let [temp (util/horizontal-matrix-concatenation inputs bias-node-matrix)
        net (inner-product temp 
                           weight-matrix)
        outputs (emap (:activation-func @nn-params) net)]
    ;(println "Inputs" inputs)
    ;(println "Outputs" outputs)
    [outputs net]))

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
  [data-set weight-matrix]
  (let [; Store the result of feeding the data-set into the network
        ff-result (feed-forward (array (:inputs data-set))
                                weight-matrix
                                (:bias data-set))
        ; Pull out the outputs from the feed forward result
        outputs (first ff-result)
        ; Pull out the resulting network from the feed forward.
        net (second ff-result)
        ; Calcuate the error of the network on the data set.
        regression-error (/ (util/sum-all-2D-matrix-components (emap square
                                                                     (- outputs
                                                                        (:outputs data-set))))
                            (* (:count data-set)
                               (count (first weight-matrix))))
        ; Find the classes of the outputs.
        classes (vec (map util/output->class outputs))
        ; Uses outputs to find the classification error on the data set.
        classification-error (do
                               ;(println "Classes:" classes)
                               ;(println "Target Classes:" (:classes data-set))f
                               (/ (count (filter true? (emap not= classes (flatten (:classes data-set)))))
                                  (count (:outputs data-set))))]
    (if (:debug-prints @nn-params)
      (do
        (println "Evaluation Of Network")
        ;(println "weights" (util/matrix-2d-pretty-print weight-matrix))
        (println "Outputs")
        (util/matrix-2d-pretty-print outputs)
        (println "Net")
        (util/matrix-2d-pretty-print net)
        (println "Regression Error" regression-error)
        (println "classification-error" classification-error)
        (println)))
    [regression-error classification-error]))

(def temp-ws 
  [[-0.478326  -0.423382  -0.083957]
   [-0.458601  -0.486521   0.390219]
   [-0.249674  -0.319713   0.089447]
   [-0.425611   0.206993   0.367458]
   [0.403078   0.053084   0.128577]])
(def foo {:count 5
          :bias (array [[1]
                        [1]
                        [1]
                        [1]
                        [1]])
          :inputs (array [[0.282131   0.269205   0.430645   0.427485]
                          [0.582374   0.445654   0.581855   0.536988]
                          [0.466896   0.480944   0.566734   0.500487]
                          [0.744044   0.516233   0.672581   0.609990]
                          [0.212844   0.516233   0.158468   0.135476]])
          :outputs (array [[0 1 0]
                           [0 1 0]
                           [0 1 0]
                           [0 1 0]
                           [1 0 0]])
          :classes (array [[2]
                           [2]
                           [2]
                           [2]
                           [1]])})
;(evaluate-network foo temp-ws)

(defn backpropagation 
  "I don't do anything right now. Come back later."
  [inputs-matrix outputs-matrix weight-matrix learning-rate bias-vector]
  (let [; Pick random sample 
        rand-sample-index (rand-int (:input-count (:data-sets @nn-params)))
        ; Feed sample through network
        ff-result (feed-forward [(nth inputs-matrix rand-sample-index)]
                                weight-matrix
                                bias-vector)
        output (first ff-result)
        net (second ff-result)
        ; Take vector of how far off feed foward was
        error-vector (- (transpose (nth outputs-matrix rand-sample-index))
                        output)
        ; How should the weight change
        delta (emap *
                    error-vector
                    (emap (:activation-func-derivative @nn-params) net))
        ; Put in temp for code readability
        temp (concat (nth inputs-matrix rand-sample-index)
                     (nth bias-vector rand-sample-index))
        ; How much should the weights change
        weights-delta (* learning-rate
                         (outer-product temp
                                        delta))
        ; What should the new weights be
        new-weights (emap + weight-matrix 
                          (reshape weights-delta 
                                   (shape weight-matrix)))]
    (if (:debug-prints @nn-params)
      (do
        (println "Backpropagation" )
        (println "w" weight-matrix)
        (println output)
        (println net)
        (println error-vector)
        (println weights-delta)
        (println new-weights)
        (println)))
    new-weights))

(defn train-nn
  [data-sets]
  (let [; Inital randomized weights to the network
        init-weight-matrix (generate-initial-weight-matrix (:max-weight-initial @nn-params)
                                                           (inc (:input-count data-sets))
                                                           (:output-count data-sets))]
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
      (if (or (= epoch (:max-epochs @nn-params))
              (if (not (empty? validation-error))
                (< (last validation-error) (:target-error-threshold @nn-params))
                false))
        (do
          (println "Training finished. Now draw some graphs.")
          (report/plot-nn-evaluations training-error
                                      training-classification-error
                                      testing-error
                                      testing-classification-error
                                      validation-error
                                      validation-classification-error))
        (let [training-eval (evaluate-network (:training-set data-sets) weights)
              testing-eval (evaluate-network (:testing-set data-sets) weights)
              validation-eval (evaluate-network (:validation-set data-sets) weights)]
          (println "Starting Epoch" (inc epoch))
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
                             (:bias (:training-set data-sets)))
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