(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.dataIO :as dIO]
            [incanter.core :as incntr]
            [incanter.charts :as charts]
            [cloann.transfer-functions :as transfer-funcs]
            [clojure.math.combinatorics :as combo])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

;(def bar [[:H2 :O] [:H1 :H2] [:I :H1]])
;(util/sort-layer-connections bar)

(def wm [[1 2]
         [3 4]
         [5 6]])

(def e [0.1 0.5])

; Looking for a vector of 3 elements

(def deltas (map (fn [x] (apply + x))
                 (map * wm (repeat e))))

(def eta 0.1)

(def 