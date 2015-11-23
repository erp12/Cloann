(ns cloann.dataIO
  (:require [cloann.util :as util]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(def empty-data-sets
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
  (let [new-matrix (array
                     (with-open [in-file (io/reader filename)]
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
(defn create-data-sub-set-from-sampled-matrix
  "Puts random sub-set of a matrix into a data sub-set map."
  [matrix data-subset-count input-indexes output-indexes classes bias]
  (let [sampled-rows (repeatedly data-subset-count #(rand-nth matrix))
        outputs (map #(vec (util/filter-by-index % %2)) sampled-rows (repeat output-indexes))]
    (-> {}
      (assoc :count data-subset-count)
      (assoc :bias (array (map vector (vec (take data-subset-count bias)))))
      (assoc :inputs (array (vec (map #(vec (util/filter-by-index % %2)) sampled-rows (repeat input-indexes)))))
      (assoc :outputs (array (vec outputs)))
      (assoc :classes (array (map vector (vec (map util/output->class outputs))))))))

(defn create-data-sub-set-from-full-matrix
  "Puts random sub-set of a matrix into a data sub-set map."
  [matrix input-indexes output-indexes classes bias]
  (let [outputs (map #(vec (util/filter-by-index % %2)) matrix (repeat output-indexes))]
    (-> {}
      (assoc :count (first (shape matrix)))
      (assoc :bias (array (map vector 
                               (vec (take (first (shape matrix)) bias)))))
      (assoc :inputs (array (vec (map #(vec (util/filter-by-index % %2)) 
                                      matrix 
                                      (repeat input-indexes)))))
      (assoc :outputs (vec outputs))
      (assoc :classes (array (map vector (vec (map util/output->class outputs))))))))

(defn create-data-sets-from-1-matrix
  "Creates a data set from one single matrix."
  [matrix input-indexes output-indexes training-count testing-count validation-count]
  (-> empty-data-sets
    (assoc :input-count (count input-indexes))
    (assoc :output-count (count output-indexes))
    (assoc :training-set (create-data-sub-set-from-sampled-matrix matrix
                                                                  training-count
                                                                  input-indexes
                                                                  output-indexes
                                                                  (vec (range (count output-indexes)))
                                                                  (vec (repeat training-count 1))))
    (assoc :testing-set (create-data-sub-set-from-sampled-matrix matrix
                                                                 testing-count
                                                                 input-indexes
                                                                 output-indexes
                                                                 (vec (range (count output-indexes)))
                                                                 (vec (repeat testing-count 1))))
    (assoc :validation-set (create-data-sub-set-from-sampled-matrix matrix
                                                                    validation-count
                                                                    input-indexes
                                                                    output-indexes
                                                                    (vec (range (count output-indexes)))
                                                                    (vec (repeat validation-count 1))))))

(defn create-data-sets-from-3-matrices
  [training-matrix testing-matrix validation-matrix input-indexes output-indexes]
  (-> empty-data-sets
    (assoc :input-count (count input-indexes))
    (assoc :output-count (count output-indexes))
    (assoc :training-set (create-data-sub-set-from-full-matrix training-matrix
                                                               input-indexes
                                                               output-indexes
                                                               (vec (range (count output-indexes)))
                                                               (repeat 1)))
    (assoc :testing-set (create-data-sub-set-from-full-matrix testing-matrix
                                                              input-indexes
                                                              output-indexes
                                                              (vec (range (count output-indexes)))
                                                              (repeat 1)))
    (assoc :validation-set (create-data-sub-set-from-full-matrix validation-matrix
                                                                 input-indexes
                                                                 output-indexes
                                                                 (vec (range (count output-indexes)))
                                                                 (repeat 1)))))