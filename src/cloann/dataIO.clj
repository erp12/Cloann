(ns cloann.dataIO
  (:require [cloann.util :as util]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(def empty-data-set
  {:input-count 0
   :output-count 0
   :training-set {:inputs []
                  :outputs []
                  :classes []
                  :count 0
                  :bias []}
   :validation-set {:inputs []
                    :outputs []
                    :classes []
                    :count 0
                    :bias []}
   :test-set {:inputs []
              :outputs []
              :classes []
              :count 0
              :bias []}})

(defn string->number [s]
  "Found on: http://stackoverflow.com/questions/10752659/how-to-convert-a-numeric-string-to-number-decimal-and-number-to-string"
  (let [n (read-string s)]
    (if (number? n) n nil)))

(defn csv->matrix
  "Reads a CSV file, parses values into doubles, puts values in a matrix.
If is-first-row-labels is true, excludes the first row."
  [filename is-first-row-labels]
  (let [new-matrix (matrix
                      (with-open [in-file (io/reader (str "data/" filename))]
                        (doall
                          (vec
                            (->> (csv/read-csv in-file)
                              (map (fn [line]
                                     (vec 
                                       (map string->number line)))))))))]
    (if is-first-row-labels 
      (rest new-matrix)
      new-matrix)))

(defn matrix->csv
  "Writes a matrix to csv file."
  [matrix filename]
  (with-open [out-file (io/writer (str "dataOut/" filename))]
    (csv/write-csv out-file matrix)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Creating Data Sets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn create-data-sub-set-from-matrix
  "Puts random sub-set of a matrix into a data sub-set map."
  [matrix data-subset-count input-indexes output-indexes classes bias]
  (let [sampled-rows (repeatedly data-subset-count #(rand-nth matrix))
        outputs (map #(vec (util/filter-by-index % %2)) sampled-rows (repeat output-indexes))]
    (-> {}
      (assoc :count data-subset-count)
      (assoc :bias (take data-subset-count bias))
      (assoc :inputs (map #(vec (util/filter-by-index % %2)) sampled-rows (repeat input-indexes)))
      (assoc :outputs outputs)
      (assoc :classes (map outputs)))))

(defn create-data-set-from-matrix
  "Creates a data set from one single matrix.  NOT DONE YET!!!"
  [matrix input-indexes output-indexes training-count testing-count validation-count]
  (-> empty-data-set
    (assoc :input-count (count input-indexes))
    (assoc :output-count (count output-indexes))
    (assoc :training-set (create-data-sub-set-from-matrix matrix
                                                          training-count
                                                          input-indexes
                                                          output-indexes
                                                          (vec (range (count output-indexes)))
                                                          (repeat 1)))
    (assoc :testing-set (create-data-sub-set-from-matrix matrix
                                                          testing-count
                                                          input-indexes
                                                          output-indexes
                                                          (vec (range (count output-indexes)))
                                                          (repeat 1)))
    (assoc :validation-set (create-data-sub-set-from-matrix matrix
                                                          validation-count
                                                          input-indexes
                                                          output-indexes
                                                          (vec (range (count output-indexes)))
                                                          (repeat 1)))))