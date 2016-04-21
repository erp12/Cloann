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
  [matrix topology-encoding [from-id to-id]]
  ;(println "gcmbi")
  ;(println matrix)
  ;(println layers)
  ;(println from-id)
  ;(println to-id)
  ;(println)
  (try
    (let [layer-ids (:layer-order-in-weight-matrix topology-encoding)
          start-row (reduce +
                            (map #(:num-nodes (% (:layers topology-encoding)))
                                 (first (split-at (.indexOf layer-ids from-id) 
                                                  layer-ids))))
          start-col (reduce +
                            (map #(:num-nodes (% (:layers topology-encoding)))
                                 (first (split-at (.indexOf layer-ids to-id) 
                                                  layer-ids))))
          height (:num-nodes (from-id (:layers topology-encoding)))
          width (:num-nodes (to-id (:layers topology-encoding)))
          result (sub-matrix matrix 
                             start-row 
                             start-col 
                             height
                             width)]
      (vec (doall result)))
    (catch Exception e
      (let [dateString (str (get-date-time-string))
            dir (str "gelnn_error_logs/")]
        (.mkdir (java.io.File. "nn_reporting/"))
        (spit (str dir dateString "_" (rand) ".txt")
              topology-encoding)
        (println (str "I caught an exception: " (.getMessage e)))))))

(defn get-connection-matrix-by-id-with-bias
  "Same as get-connection-matrix-by-id, but includes row of relevant bias nodes."
  [matrix topology-encoding [from-id to-id]]
  (let [connection-matrix-without-bias (get-connection-matrix-by-id matrix 
                                                                    topology-encoding
                                                                    [from-id to-id])
        all-bias-weights (last matrix)
        bias-for-connection (let [layer-ids (:layer-order-in-weight-matrix topology-encoding)
                                  start-col (reduce +
                                                    (map #(:num-nodes (% (:layers topology-encoding)))
                                                         (first (split-at (.indexOf layer-ids to-id) 
                                                                          layer-ids))))
                                  width (:num-nodes (to-id (:layers topology-encoding)))]
                              (subvec all-bias-weights start-col (+ start-col width)))
        result (concat connection-matrix-without-bias
                       [bias-for-connection])]
    (vec (doall result))))

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

(defn replace-layer-connection-weights-in-matrix
  "Replaces a sub-matrix inside the weight matrix."
  [matrix layers order-of-layers [from-id to-id] new-mat]
  (let [layer-ids order-of-layers
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
      (if (= r 
             (+ start-row
                height))
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

(defn replace-layer-connection-weights-and-bias-in-matrix
  "Replaces a sub-matrix inside the weight matrix and the
corrisponding bias weights."
  [matrix layers order-of-layers [from-id to-id] new-mat]
  (let [result (replace-layer-connection-weights-in-matrix matrix
                                                           layers 
                                                           order-of-layers
                                                           [from-id to-id] 
                                                           (vec butlast new-mat))
        layer-ids order-of-layers
        start-col (reduce +
                          (map #(:num-nodes (% layers))
                               (first (split-at (.indexOf layer-ids to-id) 
                                                layer-ids))))
        width (:num-nodes (to-id layers))
        result (assoc result
                      (dec (count result))
                      (loop [i start-col
                             b-vec (last result)]
                        (if (< i (+ start-col width))
                          (recur (inc i)
                                 (assoc b-vec
                                        i
                                        (get (last new-mat)
                                             i)))
                          b-vec)))]
    result))

(defn num-nodes-in-network
  "Based on layers encoding, returns number of nodes in the network."
  [layers]
  (apply +
         (map #(:num-nodes %) 
              (vals layers))))

(defn in? 
  "true if coll contains elm.
http://stackoverflow.com/questions/3249334/test-whether-a-list-contains-a-specific-value-in-clojure"
  [coll elm]  
  (some #(= elm %) coll))

(defn sort-layer-connections
  "Sorts layer connections so that signal begins at input layer and is passed
sequencially through layers."
  [layer-conns]
  (loop [remaining-layer-conns layer-conns
         sorted-layer-conns []
         looking-for-connections-from [:I]]
    (let [found-layers (filter #(in? looking-for-connections-from
                                     (first %)) 
                               remaining-layer-conns)]
      (cond
        (empty? remaining-layer-conns); All the layers have been sorted
        sorted-layer-conns
        (empty? looking-for-connections-from); Useless layer has been found, take it out! (THIS SHOULD MOVE)
        (do
          (println "That code that shouldn't run just ran, but its cool. [FORWARD]")
          sorted-layer-conns)
        :else
        (recur (remove #(in? looking-for-connections-from
                             (first %))
                       remaining-layer-conns)
               (concat sorted-layer-conns
                       found-layers)
               (vec (map #(second %) found-layers)))))))

(defn sort-layer-connections-back
  "Sorts layer connections so that signal begins at output layer and is passed
sequencially through layers, backwards."
  [layer-conns]
  (loop [remaining-layer-conns layer-conns
         sorted-layer-conns []
         looking-for-connections-from [:O]]
    (let [found-layers (filter #(in? looking-for-connections-from
                                     (second %)) 
                               remaining-layer-conns)]
      (cond
        (empty? remaining-layer-conns); All the layers have been sorted
        sorted-layer-conns
        (empty? looking-for-connections-from); Useless layer has been found, take it out! (THIS SHOULD MOVE)
        (do
          (println "That code that shouldn't run just ran, but its cool. [BACK]")
          sorted-layer-conns)
        :else
        (recur (remove #(in? looking-for-connections-from
                             (second %))
                       remaining-layer-conns)
               (concat sorted-layer-conns
                       found-layers)
               (vec (map #(first %) found-layers)))))))

(defn remove-useless-topology-from-encoding
  "Removed layer-connections and layers from a network encoding that will have
no effect on training."
  [old-topology-encoding]
  ;(println "Start toplogy simplify")
  (let [removed-useless-connections (assoc old-topology-encoding
                                           :layer-connections
                                           (vec
                                             (loop [remaining-layer-conns (:layer-connections old-topology-encoding)
                                                    new-layer-conns []
                                                    looking-for-connections-to [:O]]
                                               (let [found-layers-conns (filter #(in? looking-for-connections-to
                                                                                      (second %)) 
                                                                                remaining-layer-conns)]
                                                 (cond
                                                   (empty? remaining-layer-conns)
                                                   new-layer-conns
                                                   (empty? looking-for-connections-to)
                                                   new-layer-conns
                                                   :else
                                                   (recur (remove #(in? looking-for-connections-to
                                                                        (second %))
                                                                  remaining-layer-conns)
                                                          (concat new-layer-conns
                                                                  found-layers-conns)
                                                          (vec (map #(first %) 
                                                                    found-layers-conns))))))))
        ;foo (println "Done connection simplify")
        removed-useless-layers (assoc removed-useless-connections
                                      :layers
                                      (into {}
                                            (filter (fn [l]
                                                      (if (in? (flatten (:layer-connections removed-useless-connections))
                                                               (first l))
                                                        true
                                                        false))
                                                    (:layers removed-useless-connections))))
        ;foo (println "Done layers simplify")
        ]
    (assoc removed-useless-layers
           :layer-connections
           (vec (:layer-connections removed-useless-layers)))))
