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

(defn foo
  [l]
  (+ l [10 20]))

(def wm [[1 2]
         [3 4]
         [5 6]])


(apply foo wm)
