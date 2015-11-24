(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO]
            [incanter.core :as incntr]
            [incanter.charts :as charts])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(defn generate-initial-weight-matrix
  "Generates matrix of weights between max-weight and negative max-weight"
  [max-weight num-inputs num-outputs num-hidden]
  ; Save the size of the weight matrix
  (let [weight-matrix-width (+ num-inputs
                               num-outputs
                               num-hidden)]
    ; Start with an empty matrix, and populate from there.
    (loop [weight-matrix [[]]]
      ; if the weight-matrix does not have enough rows
      (if (< (count weight-matrix) 
             weight-matrix-width)
        ; if the last row of the matrix does not have all its columns
        (if (< (count (last weight-matrix)) 
               weight-matrix-width)
          ; recur the loop with a new weight added to the end of the last row
          (recur (assoc-in weight-matrix
                           [(dec (count weight-matrix)) (count (last weight-matrix))]
                           ; if either the row or the column are not related to an input...
                           (if (or (> (count weight-matrix)
                                      num-inputs)
                                   (> (inc (count (last weight-matrix)))
                                      num-inputs))
                             ; generate a random weight
                             (- (rand (* max-weight 2))
                              max-weight)
                             ; otherwise, no connection
                             nil)))
          ; (if last row has alll its weights) append a new row on to the matrix
          (recur (conj weight-matrix [])))
        ; (if the weight matrix has enough rows) and if the last row does not have all its weights yet.
        (if (< (count (last weight-matrix)) 
               weight-matrix-width)
          ; recur the loop with a new weight added to the end of the last row
          (recur (assoc-in weight-matrix
                           [(dec (count weight-matrix)) (count (last weight-matrix))]
                           ; if either the row or the column are not related to an input (should only be true here with 0 hidden nodes)
                           (if (or (> (count weight-matrix) 
                                      num-inputs)
                                   (> (inc (count (last weight-matrix)))
                                      num-inputs))
                             ; generate a random weight
                             (- (rand (* max-weight 2))
                              max-weight)
                             ; otherwise, no connection
                             nil)))
          weight-matrix)))))

(util/matrix-2d-pretty-print (generate-initial-weight-matrix 0.5 3 2 0))