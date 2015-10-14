(ns cloann.matrix-testing
  (:gen-class)
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(def M1 (matrix [[1 2][3 4]]))
(apply + (apply + M1))