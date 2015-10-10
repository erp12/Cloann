(ns cloann.core
  (:gen-class)
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(defn feed-forward
  [input-matrix weight-matrix bias-node-matrix]
  (let [net
        output]
    ()))