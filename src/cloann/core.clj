(ns cloann.core
  (:gen-class)
  (:use [cloann util activation-functions])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(def nn-params 
  (atom 
    {:activation-func hyperbolic-tangent
     :activation-func-derivative hyperbolic-tangent-derivative
     :max-weight 1}))

(defn feed-forward
  [input-matrix weight-matrix bias-node-matrix]
  (let [net (* weight-matrix
               (horizontal-matrix-concatenation input-matrix bias-node-matrix))
        output (emap (:activation-func @nn-params) net)]
    ;; OUTPUT value not correct. Must check what activate() does from blog post.
    [net output]))

(defn generate-initial-weight-matrix
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
  [input-matrix weight-matrix target-output-matrix target-class-matrix bias-matrix]
  ())

(defn -main 
  []
  (println "I don't do anything yet..."))