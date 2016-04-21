(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.core :as cloann]
            [cloann.util :as util]
            [cloann.dataIO :as dIO]
            [incanter.core :as incntr]
            [incanter.charts :as charts]
            [cloann.transfer-functions :as transfer-funcs]
            [clojure.math.combinatorics :as combo])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

;(def bar [[:H2 :O] [:H1 :H2] [:I :H1]])
;(util/sort-layer-connections bar)

(def wm [[nil nil 0]
         [nil 0 nil]
         [0 nil 0]])


(cloann/initialize-weights wm 1)



