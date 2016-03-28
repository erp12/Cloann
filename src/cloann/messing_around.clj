(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO]
            [incanter.core :as incntr]
            [incanter.charts :as charts]
            [clojure.math.combinatorics :as combo])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(def layers  {:I {:num-inputs 4
                  :num-outputs 5}
              :H1 {:num-inputs 5
                   :num-outputs 5}
              :H2 {:num-inputs 5
                   :num-outputs 4}
              :O {:num-inputs 4
                  :num-outputs 3}})
(def layer-conns [[:I :H1]
                  [:H1 :H2]
                  [:H2 :H1]
                  [:H2 :O]])

(def uninitialized-weights [[nil nil nil nil nil 0   1   2   3   4   nil nil nil nil nil nil nil]
                            [nil nil nil nil nil 5   6   7   8   9   nil nil nil nil nil nil nil]
                            [nil nil nil nil nil 10  11  12  13  14  ":)" nil nil nil nil nil nil]
                            [nil nil nil nil nil 15  16  17  18  19  nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 20  21  22  23  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 24  25  26  27  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 28  29  30  31  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 32  33  34  35  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 36  37  38  39  nil nil nil]
                            [nil nil nil nil nil 40  41  42  43  44  nil nil nil nil 65  66  67 ]
                            [nil nil nil nil nil 45  46  47  48  49  nil nil nil nil 68  69  70 ]
                            [nil nil nil nil nil 50  51  52  53  54  nil nil nil nil 71  72  73 ]
                            [nil nil nil nil nil 55  56  57  58  59  nil nil nil nil 74  75  76 ]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]])

(def M2 [["a" "b" "c" "d" "u"]
         ["e" "f" "g" "h" "v"]
         ["i" "j" "k" "l" "w"]
         ["m" "n" "o" "p" "x"]])

(def M3 [["a" "b" "c"]
         ["e" "f" "g"]
         ["i" "j" "k"]
         ["m" "n" "o"]])

;(util/matrix-2d-pretty-print (util/replace-layer-connection uninitialized-weights 
;                                                            layers 
;                                                            [:I :H1] 
;                                                            M2))
;(println)
;(util/matrix-2d-pretty-print (util/replace-layer-connection uninitialized-weights 
;                                                            layers 
;                                                            [:H2 :O] 
;                                                            M3))

;(util/matrix-2d-pretty-print uninitialized-weights )
