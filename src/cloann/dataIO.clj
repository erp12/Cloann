(ns cloann.dataIO
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

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