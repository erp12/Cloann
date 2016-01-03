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

(def M1 [[:ii :ii :ii :ii :ii :ih :ih :ih :ih :io :io :io]
         [:ii :ii :ii :ii :ii :ih :ih :ih :ih :io :io :io]
         [:ii :ii :ii :ii :ii :ih :ih :ih :ih :io :io :io]
         [:ii :ii :ii :ii :ii :ih :ih :ih :ih :io :io :io]
         [:hi :hi :hi :hi :hi :hh :hh :hh :hh :ho :ho :ho]
         [:hi :hi :hi :hi :hi :hh :hh :hh :hh :ho :ho :ho]
         [:hi :hi :hi :hi :hi :hh :hh :hh :hh :ho :ho :ho]
         [:hi :hi :hi :hi :hi :hh :hh :hh :hh :ho :ho :ho]
         [:hi :hi :hi :hi :hi :hh :hh :hh :hh :ho :ho :ho]
         [:oi :oi :oi :oi :oi :oh :oh :oh :oh :oo :oo :oo]
         [:oi :oi :oi :oi :oi :oh :oh :oh :oh :oo :oo :oo]
         [:oi :oi :oi :oi :oi :oh :oh :oh :oh :oo :oo :oo]
         [:oi :oi :oi :oi :oi :oh :oh :oh :oh :oo :oo :oo]])

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

(def uninitialized-weights [["a" nil nil nil nil 0   1   2   3   4   "b" nil nil nil "c" nil nil]
                            [nil nil nil nil nil 5   6   7   8   9   nil nil nil nil nil nil nil]
                            [nil nil nil nil nil 10  11  12  13  14  nil nil nil nil nil nil nil]
                            [nil nil nil nil nil 15  16  17  18  19  nil nil nil nil nil nil nil]
                            ["d" nil nil nil nil "e" nil nil nil nil 20  21  22  23  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 24  25  26  27  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 28  29  30  31  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 32  33  34  35  nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil 36  37  38  39  nil nil nil]
                            [nil nil nil nil nil 40  41  42  43  44  nil nil nil nil 65  66  67 ]
                            [nil nil nil nil nil 45  46  47  48  49  nil nil nil nil 68  69  70 ]
                            [nil nil nil nil nil 50  51  52  53  54  nil nil nil nil 71  72  73 ]
                            [nil nil nil nil nil 55  56  57  58  59  nil nil nil nil 74  75  76 ]
                            [nil nil nil nil nil 60  61  62  63  64  nil nil nil nil 77  78  79 ] 
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]
                            [nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil]])


(util/matrix-2d-pretty-print (util/get-connection-matrix-by-id uninitialized-weights layers [:I :H2]))
