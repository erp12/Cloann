(ns cloann.util
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(defn nat-log
  "Implements LN function using Math/log"
  [x]
  (/ (Math/log x) (Math/log Math/E)))

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
    ;(matrix
      (vec
        (map (fn [class-num]
               (assoc (vec (take (inc vector-len) (repeat 0)))
                      (dec class-num)
                      1))
             class-num-vec))))

(defn horizontal-matrix-concatenation
  "Same as horzcat() function from MATLAB."
  [matrix-1 matrix-2]
  (array (vec (map concat (rows matrix-1) (rows matrix-2)))))

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
  [matrix [from-id to-id]]
  (let [layers-info (meta matrix)
        layer-ids (keys layers-info)
        start-row (reduce +
                          (map #(:num-inputs (% layers-info))
                               (first (split-at (.indexOf layer-ids from-id) 
                                                layer-ids))))
        start-col (reduce +
                          (map #(:num-outputs (% layers-info))
                               (first (split-at (.indexOf layer-ids to-id) 
                                                layer-ids))))
        height (:num-inputs (from-id layers-info))
        width (:num-inputs (from-id layers-info))]
    (sub-matrix matrix start-row start-col height width)))

(defn create-2D-vector-of-val
  "Creates vector of vectors of given dimensions where every value is x."
  [rows cols x]
  (vec (repeat rows
               (vec (repeat cols
                            x)))))