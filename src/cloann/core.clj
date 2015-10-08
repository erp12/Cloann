(ns cloann.core
  (:gen-class)
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(defn output->class
  "Converts the output matrix of the ANN to a single number per observation,
corrispoding to that observation's class"
  [output-matrix]
  (map (fn [v]
         (.indexOf v 1))
       output-matrix))

(defn class->output
  "Converts a vector of class numbers, into the desired ANN output matrix"
  [class-num-vec]
  (let [vector-len (apply max class-num-vec)]
    ;(matrix
      (vec
        (map (fn [class-num]
               (assoc (vec (take (inc vector-len) (repeat 0)))
                      class-num
                      1))
             class-num-vec))))