(ns cloann.core
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(def nn-params
  (atom 
    {:activation-func act-funcs/hyperbolic-tangent
     :activation-func-derivative act-funcs/hyperbolic-tangent-derivative
     :max-weight 1
     :data-set nil}))

(defn feed-forward
  [input-matrix weight-matrix bias-node-matrix]
  (let [net (* weight-matrix
               (util/horizontal-matrix-concatenation input-matrix bias-node-matrix))
        output (emap (:activation-func @nn-params) net)]
    ;; OUTPUT value may not correct. Unclear what activate() does from blog post.
    [net output]))

(defn generate-initial-weight-matrix
  "Random weight matrix."
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
  [data-set weight-matrix]
  (let [ff-result (feed-forward input-matrix weight-matrix bias-matrix)
        output (first ff-result)
        net (second ff-result)
        error (/ (util/sum-all-2D-matrix-components (Math/pow (- (:outputs data-set)
                                                                 output)
                                                              2))
                 (* (count (:outputs data-set)) (:output-count data-set)))
        classes (util/output->class output)
        classification-error (/ (count (filter true? (map = classes (:classes data-set)))))
                                (count (:outputs data-set))]
    [error c]))

(defn backpropagation 
  []
  ())

(defn -main 
  []
  (println "I don't do anything yet..."))