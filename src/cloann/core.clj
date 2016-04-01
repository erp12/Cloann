(ns cloann.core
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.transfer-functions :as transfer-funcs]
            [cloann.nn-reporting :as report]
            [clojure.math.combinatorics :as combo])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

; Default params to the neural network.
(def nn-params
  (atom 
    {:transfer-func transfer-funcs/hyperbolic-tangent
     :transfer-func-derivative transfer-funcs/hyperbolic-tangent-derivative
     :max-weight-initial 1
     :data-set nil ; Must be set in problem file
     :topology-encoding nil ; Must be set in problem file
     :plot-graphs true
     :learning-rate 0.02
     :max-epochs 500
     :debug-prints false
     :validation-stop-threshold 0.03}))

(defn generate-sub-matrices-from-layer-data
  "Returns uninitialized weight matrices for each inter-layer connections.
0 are uninitialized weights, and nils are no connection."
  [layers layer-conns]
  (let [distinct-ids (distinct (flatten layer-conns))
        all-id-pairs (combo/selections distinct-ids 2)]
    (partition
      (count distinct-ids)
      (for [ids all-id-pairs]
        (let [i ((first ids) layers)
              o ((second ids) layers)]
          (util/create-2D-vector-of-val (:num-nodes i)
                                        (:num-nodes o)
                                        (if (some #(= ids %) 
                                                  layer-conns)
                                          0
                                          nil)))))))

(defn generate-uninitialized-weight-matrix
  "Returns entire network matrix with uninitialized weights between layers
that are connected."
  [layers layer-conns]
  (let [sub-weight-matries (generate-sub-matrices-from-layer-data layers 
                                                                  layer-conns)]
    (concat
      (reduce concat
              (map #(reduce util/horizontal-matrix-concatenation 
                            %)
                   sub-weight-matries))
      [(vec (repeat (util/num-nodes-in-network (:layers (:topology-encoding @nn-params)))
                    0))])))

(defn initialize-weights
  "replaces all 0s in the matrix with random weight w where -max-w < w < max-w.
Leaves the nils in the matrix."
  [matrix max-w]
  (clojure.walk/walk
    (fn [x]
      (clojure.walk/walk #(if (= 0 %)
                            (- (rand (* 2 
                                        max-w))
                               max-w))
                         #(vec %)
                         x))
    (fn [x] (vec x))
    matrix))

(defn feed-forward-layer
  "Computes the output of a single connection between layers of the 
neural network given the inputs and the weights."
  [inputs weight-matrix]
  (let [net (inner-product (conj inputs 1)
                           weight-matrix)
        outputs (emap (:transfer-func @nn-params) net)]
    ;(println net)
    {:output outputs 
     :sum-of-weighted-inputs net}))

(defn feed-forward
  "Takes input values and returns the outputs based on network weights."
  [input-values weight-matrix]
  (loop [remaining-layer-connections (util/sort-layer-connections (:layer-connections (:topology-encoding @nn-params)))
         current-inputs input-values
         result {:output nil
                 :sum-of-weighted-inputs {}
                 :ff-order []}]
    (if (empty? remaining-layer-connections)
      (assoc result :output current-inputs)
      (let [ff-layer-result (feed-forward-layer current-inputs
                                                (util/get-connection-matrix-by-id-with-bias weight-matrix
                                                                                            (:layers (:topology-encoding @nn-params))
                                                                                            (first remaining-layer-connections)))]
        (recur (rest remaining-layer-connections)
               (:output ff-layer-result)
               (assoc-in (assoc result
                                :ff-order
                                (concat (:ff-order result)
                                        [(first remaining-layer-connections)]))
                         [:sum-of-weighted-inputs (second (first remaining-layer-connections))]
                         (:sum-of-weighted-inputs ff-layer-result)))))))



(defn evaluate-network
  "Returns the error, and classification error of the network on a particular data-set"
  [data-set weight-matrix]
  (let [; Store the result of feeding the data-set into the network
        ff-result (feed-forward (array (:inputs data-set))
                                weight-matrix
                                (:bias data-set))
        ; Pull out the outputs from the feed forward result
        outputs (first ff-result)
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
        classification-error (/ (count (filter true? (emap not= classes (flatten (:classes data-set)))))
                                (count (:outputs data-set)))]
    (if (:debug-prints @nn-params)
      (do
        (println "Evaluation Of Network")
        (println "Outputs")
        (util/matrix-2d-pretty-print outputs)
        (println "Net")
        (util/matrix-2d-pretty-print net)
        (println "Regression Error" regression-error)
        (println "Classes" classes)
        (println "classification-error" classification-error)
        (println)))
    [regression-error classification-error]))

(defn error-signal-of-hidden-layer
  ""
  [weight-matrix-from-this-layer errors-of-next-layer]
  (vec
    (map (fn [x] (apply + x))
         (map * 
              weight-matrix-from-this-layer 
              (repeat errors-of-next-layer)))))

(defn backpropagation 
  "Returns the new weights for the network after 1 step of back propagation."
  [input-pattern output-pattern weight-matrix learning-rate]
  (let [; Feed sample through network
        ff-result (feed-forward input-pattern
                                weight-matrix)
        error-of-output (- output-pattern 
                           (:output ff-result))
        error-signals (loop [remaining-layer-connections-back (reverse (:ff-order ff-result))
                             deltas {:O error-of-output}]
                        (if (empty? remaining-layer-connections-back)
                          deltas
                          (recur (rest remaining-layer-connections-back)
                                 (assoc deltas
                                        (first (first remaining-layer-connections-back))
                                        (error-signal-of-hidden-layer (util/get-connection-matrix-by-id weight-matrix
                                                                                                        (:layers (:topology-encoding @nn-params))
                                                                                                        (first remaining-layer-connections-back))
                                                                      (get deltas
                                                                           (second (first remaining-layer-connections-back))))))))
        ]
    (println "Error Signals: " error-signals)
    ))


;(defn train-nn
;  [data-sets]
;  (let [; Inital randomized weights to the network
;        init-weight-matrix (generate-initial-weight-matrix (:max-weight-initial @nn-params)
;                                                           (inc (:input-count data-sets))
;                                                           (:output-count data-sets))
;        ; inputs to the network
;        inputs (:inputs (:training-set data-sets))
;        outputs (:outputs (:training-set data-sets))
;        bias (:bias (:training-set data-sets))]
;    (loop [; nn's training epoch
;           epoch 0
;           ; current set of weights for the network
;           weights init-weight-matrix
;           ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;           ; Tracking info for reporting later
;           training-error []
;           training-classification-error []
;           testing-error []
;           testing-classification-error []
;           validation-error []
;           validation-classification-error []]
;      (let [max-epochs-reached? (= epoch 
;                                   (:max-epochs @nn-params))
;            stop-threshold-reached? (and (not (empty? validation-error))
;                                         (< (last validation-error) 
;                                            (:validation-stop-threshold @nn-params)))]
;        (if (or max-epochs-reached?
;                stop-threshold-reached?)
;          (do
;            (println "Training finished. Now draw some graphs.")
;            (if stop-threshold-reached?
;              (println "Validation Stop Threshold Reached"))
;            (if max-epochs-reached?
;              (println "Max Epochs Reached"))
;            (report/plot-nn-evaluations training-error
;                                        training-classification-error
;                                        testing-error
;                                        testing-classification-error
;                                        validation-error
;                                        validation-classification-error))
;          (let [training-eval (evaluate-network (:training-set data-sets) weights)
;                testing-eval (evaluate-network (:testing-set data-sets) weights)
;                validation-eval (evaluate-network (:validation-set data-sets) weights)]
;            (println "Starting Epoch" (inc epoch))
;            (recur 
;              ; Increment the epoch number
;              (inc epoch)
;              ; Apply the results of the backpropagation as the new weights
;              (backpropagation inputs
;                               outputs
;                               weights 
;                               (:learning-rate @nn-params)
;                               bias)
;              ;;;;;;;;;;;;;;;;;;;;;;;;
;              ; Append errors to vectors for reporting
;              (conj training-error (first training-eval))
;              (conj training-classification-error (second training-eval))
;              (conj testing-error (first testing-eval))
;              (conj testing-classification-error (second testing-eval))
;              (conj validation-error (first validation-eval))
;              (conj validation-classification-error (second validation-eval)))))))))
;
;(defn run-cloann
;  [params]
;  (swap! nn-params #(merge % params))
;  ;(println (:data-sets @nn-params))
;  (train-nn (:data-sets @nn-params)))

(defn -main 
  []
  (println "I don't do anything yet..."))