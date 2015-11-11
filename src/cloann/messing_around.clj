(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(def M1 (array [[2 4] 
                [0 1]
                [3 5]]))

(def M2 (array [[[0 3] 
                [1 2]]))

(def M3 (array [[0.2 1.5 0.7 2.3 1.1]
                [1.0 0.3 4.3 2.3 2.3]]))