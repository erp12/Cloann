(ns cloann.util
  (:require [clojure.math.combinatorics :as combo])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(defn nat-log
  "Implements LN function using Math/log"
  [x]
  (/ (Math/log x) 
     (Math/log Math/E)))

(defn get-date-time-string
  "Returns date and time.
Taken from: http://stackoverflow.com/questions/4635680/what-is-the-best-way-to-get-date-and-time-in-clojure"
  []
  (clojure.string/replace (str (new java.util.Date))
                          ":"
                          "_"))

(defn output->class
  "Converts the output matrix of the ANN to a single number per observation,
corrispoding to that observation's class"
  [output-vector]
  (let [max-in-vec (apply max output-vector)]
    (inc (.indexOf output-vector max-in-vec))))

(defn class->output
  "Converts a vector of class numbers, into the desired ANN output matrix"
  [class-num-vec]
  (let [vector-len (apply max class-num-vec)]
    (vec
      (map (fn [class-num]
             (assoc (vec (take (inc vector-len) (repeat 0)))
                    (dec class-num)
                    1))
           class-num-vec))))

(defn horizontal-matrix-concatenation
  "Same as horzcat() function from MATLAB."
  [matrix-1 matrix-2]
  (array (vec (map concat 
                   (rows matrix-1) 
                   (rows matrix-2)))))

(defn sub-matrix 
  "Returns a sub matrix that starts at row,col."
  [matrix row col height width]
  (let [rows (subvec matrix
                     row
                     (+ row height))]
    (map subvec
         rows
         (repeat col)
         (repeat (+ col width)))))



(defn sum-all-2D-matrix-components
  [matrix]
  (apply + 
         (apply + 
                matrix)))

(defn filter-by-index [coll idxs]
  "Filters out only certain indexes in a vector.
Taken from here: http://stackoverflow.com/questions/7744656/how-do-i-filter-elements-from-a-sequence-based-on-indexes"
  (keep-indexed #(when ((set idxs) %1) %2) 
                coll))

(defn matrix-2d-pretty-print
  "Prints a 2d matrix in a pretty way. Returns nil"
  [matrix]
  (doall (map #(println %) matrix))
  nil)

(defn data-set-pretty-print
  "Prints a data-set in a pretty way"
  [data-set]
  (println "Counts:" (:count data-set))
  (println "Inputs:")
  (doseq [row (:inputs data-set)]
    (println row))
  ;(doall (map println (:inputs data-set)))
  (println "Outputs:")
  (doseq [row (:outputs data-set)]
    (println row))
  (println "classes:")
  (doseq [row (:classes data-set)]
    (println row))
  (println "Bias:")
  (doseq [row (:bias data-set)]
    (println row )))

(defn get-connection-matrix-by-id
  "Returns a sub matrix for the connections between the two layers."
  [matrix layers [from-id to-id]]
  ;(println "gcmbi")
  ;(println matrix)
  ;(println layers)
  ;(println from-id)
  ;(println to-id)
  ;(println)
  (let [layer-ids (keys layers)
        start-row (reduce +
                          (map #(:num-nodes (% layers))
                               (first (split-at (.indexOf layer-ids from-id) 
                                                layer-ids))))
        start-col (reduce +
                          (map #(:num-nodes (% layers))
                               (first (split-at (.indexOf layer-ids to-id) 
                                                layer-ids))))
        height (:num-nodes (from-id layers))
        width (:num-nodes (to-id layers))]
    (sub-matrix matrix start-row start-col height width)))

(defn create-2D-vector-of-val
  "Creates vector of vectors of given dimensions where every value is x."
  [rows cols x]
  (vec (repeat rows
               (vec (repeat cols
                            x)))))

(defn all-pairs 
  "Creates list of vecots containing every pair of 2 elements in coll.
Taken from: http://stackoverflow.com/questions/4053845/idomatic-way-to-iterate-through-all-pairs-of-a-collection-in-clojure"
  [coll]
  (when-let [s (next coll)]
    (lazy-cat (for [y s] [(first coll) y])
              (all-pairs s))))

(defn replace-layer-connection
  ""
  [matrix layers [from-id to-id] new-mat]
  (let [layer-ids (keys layers)
        start-row (reduce +
                          (map #(:num-nodes (% layers))
                               (first (split-at (.indexOf layer-ids from-id) 
                                                layer-ids))))
        start-col (reduce +
                          (map #(:num-nodes (% layers))
                               (first (split-at (.indexOf layer-ids to-id) 
                                                layer-ids))))
        height (:num-nodes (from-id layers))
        width (:num-nodes (to-id layers))]
    (loop [r start-row
           c start-col
           m matrix]
      (if (and (= r 
                  (+ start-row
                     height))
               (= c
                  (+ start-col
                     width)))
        m
        (if (< c
               (+ start-col
                  width))
          (recur r
                 (inc c)
                 (assoc-in m
                           [(int r) (int c)]
                           (get-in new-mat
                                   [(int (- r
                                            start-row))
                                    (int (- c
                                            start-col))])))
          (recur (inc r)
                 start-col
                 m))))))