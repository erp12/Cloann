(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(def M1 (array [[1 2] 
                [3 2]]))

(def foo (emap act-funcs/hyperbolic-tangent
               M1))