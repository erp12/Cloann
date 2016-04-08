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
                 :ff-order []
                 :layer-outputs {:I input-values}}]
    (if (empty? remaining-layer-connections)
      (assoc result :output current-inputs)
      (let [ff-layer-result (feed-forward-layer current-inputs
                                                (util/get-connection-matrix-by-id-with-bias weight-matrix
                                                                                            (:layers (:topology-encoding @nn-params))
                                                                                            (first remaining-layer-connections)))]
        (recur (rest remaining-layer-connections)
               (:output ff-layer-result)
               (assoc-in (assoc (assoc-in result
                                          [:layer-outputs (second (first remaining-layer-connections))]
                                          (:output ff-layer-result))
                                :ff-order
                                (concat (:ff-order result)
                                        [(first remaining-layer-connections)]))
                         [:sum-of-weighted-inputs (second (first remaining-layer-connections))]
                         (:sum-of-weighted-inputs ff-layer-result)))))))

(defn evaluate-network
  "Returns the error, and classification error of the network on a particular data-set"
  [input-patterns output-patterns weight-matrix]
  (let [ff-results (map feed-forward
                        input-patterns
                        (repeat weight-matrix))
        ff-outputs-matrix (vec (map #(:output %)
                                    ff-results))
        regression-error (/ (util/sum-all-2D-matrix-components (emap square
                                                                     (- output-patterns 
                                                                        ff-outputs-matrix)))
                            (* (count output-patterns)
                               (count (first output-patterns))))
        classes (map util/output->class ff-outputs-matrix)
        ;classification-error 
        ]
    {:regresion-error regression-error
     :classification-error 0}))

(defn error-signal-of-hidden-layer
  ""
  [weight-matrix-from-this-layer errors-of-next-layer]
  (vec
    (map (fn [x] (apply + x))
         (map * 
              weight-matrix-from-this-layer 
              (repeat errors-of-next-layer)))))

(defn calculate-new-weight
  ""
  [old-weight error-signal-next-node sum-weighted-inputs-next-node output-previous-node]
  (+ old-weight
     (* (:learning-rate @nn-params)
        error-signal-next-node
        ((:transfer-func-derivative @nn-params) sum-weighted-inputs-next-node)
        output-previous-node)))

(defn new-inter-layer-connection-weights
  ""
  [old-weights to-layer-error-signals to-layer-SWI from-layer-outputs]
  (loop [new-weights old-weights
         r 0
         c 0]
    (cond 
      (>= r
          (first (shape new-weights)))
      new-weights
      (>= c
          (second (shape new-weights)))
      (recur new-weights
             (inc r)
             0)
      :else
      (recur (assoc-in new-weights
                       [r c]
                       (calculate-new-weight (get-in new-weights [r c])
                                             (nth to-layer-error-signals c)
                                             (nth to-layer-SWI c)
                                             (nth from-layer-outputs r)))
             r
             (inc c)))))

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
                                                                           (second (first remaining-layer-connections-back))))))))]
    (loop [remaining-layer-connections (:ff-order ff-result)
           new-weight-matrix weight-matrix]
      (if (empty? remaining-layer-connections)
        new-weight-matrix
        (recur
          (rest remaining-layer-connections)
          (let [current-conn (first remaining-layer-connections)
                from-layer (first current-conn)
                to-layer (second current-conn)
                current-conn-old-weights (util/get-connection-matrix-by-id new-weight-matrix
                                                                           (:layers (:topology-encoding @nn-params))
                                                                           current-conn)]
            (util/replace-layer-connection-weights-in-matrix new-weight-matrix
                                                             (:layers (:topology-encoding @nn-params))
                                                             current-conn
                                                             (new-inter-layer-connection-weights current-conn-old-weights
                                                                                                 (to-layer error-signals)
                                                                                                 (to-layer (:sum-of-weighted-inputs ff-result))
                                                                                                 (from-layer (:layer-outputs ff-result))))))))))   


(defn train-nn
  [data-sets]
  (let [; Inital randomized weights to the network
        initial-weight-matrix (initialize-weights (generate-uninitialized-weight-matrix (:layers (:topology-encoding @nn-params))
                                                                                        (:layer-connections (:topology-encoding @nn-params)))
                                                  (:max-weight-intial @nn-params))
        ; inputs to the network
        inputs (:inputs (:training-set data-sets))
        outputs (:outputs (:training-set data-sets))]
    (loop [; nn's training epoch
           epoch 0
           ; current set of weights for the network
           weights initial-weight-matrix
           ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
           ; Tracking info for reporting later
           training-error []
           training-classification-error []
           testing-error []
           testing-classification-error []
           validation-error []
           validation-classification-error []]
      (let [max-epochs-reached? (= epoch 
                                   (:max-epochs @nn-params))
            stop-threshold-reached? (and (not (empty? validation-error))
                                         (< (last validation-error)
                                            (:validation-stop-threshold @nn-params)))]
        (if (or max-epochs-reached?
                stop-threshold-reached?)
          (do
            (println "Training finished. Now draw some graphs.")
            (if stop-threshold-reached?
              (println "Validation Stop Threshold Reached"))
            (if max-epochs-reached?
              (println "Max Epochs Reached"))
            (report/plot-nn-evaluations training-error
                                        training-classification-error
                                        testing-error
                                        testing-classification-error
                                        validation-error
                                        validation-classification-error))
          (let [training-eval (evaluate-network (:inputs (:training-set data-sets))
                                                (:outputs (:training-set data-sets))
                                                weights)
                testing-eval (evaluate-network (:inputs (:testing-set data-sets))
                                               (:outputs (:testing-set data-sets))
                                               weights)
                validation-eval (evaluate-network (:inputs (:validation-set data-sets))
                                                  (:outputs (:validation-set data-sets))
                                                  weights)]
            (println "Starting Epoch" (inc epoch))
            (recur 
              ; Increment the epoch number
              (inc epoch)
              ; Apply the results of the backpropagation as the new weights
              
              (loop [wm weights
                     remaining-patterns-inds (shuffle (range (count (:inputs (:validation-set data-sets)))))]
                (if (empty? remaining-patterns-inds)
                  wm
                  (recur (backpropagation (nth (:inputs (:testing-set data-sets)) 
                                               (first remaining-patterns-inds))
                                          (nth (:outputs (:testing-set data-sets)) 
                                               (first remaining-patterns-inds))
                                          wm
                                          (:learning-rate @nn-params))
                         (rest remaining-patterns-inds))))
              ;;;;;;;;;;;;;;;;;;;;;;;;
              ; Append errors to vectors for reporting
              (conj training-error (:regresion-error training-eval))
              (conj training-classification-error (:classification-error training-eval))
              (conj testing-error (:regresion-error testing-eval))
              (conj testing-classification-error (:classification-error testing-eval))
              (conj validation-error (:regresion-error validation-eval))
              (conj validation-classification-error (:classification-error validation-eval)))))))))

(defn run-cloann
  [params]
  (swap! nn-params #(merge % params))
  ;(println (:data-sets @nn-params))
  (train-nn (:data-sets @nn-params)))

(defn -main 
  []
  (println "I don't do anything yet..."))