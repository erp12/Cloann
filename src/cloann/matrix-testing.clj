(ns cloann.matrix-testing
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(def W (matrix [[0 1 2]
                [3 4 5]
                [6 7 8]]))

(defn repeat-func-over-mat
  [init-matrix func num]
  (loop [m init-matrix
         i 0]
    (if (= i num)
      m
      (do
        (println m)
        (recur (emap func m)
               (inc i))))))