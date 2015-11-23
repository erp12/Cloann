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
  ;(println matrix-1)
  ;(println matrix-2)
  ;(println "")
  (array (vec (map concat (rows matrix-1) (rows matrix-2)))))

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
