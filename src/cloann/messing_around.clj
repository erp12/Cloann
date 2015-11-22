(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


(def M1 (array [[1 2] 
                [3 2]]))


(def foo {:count 5
          :bias (array [[1]
                        [1]
                        [1]
                        [1]
                        [1]])
          :inputs (array [[0.282131   0.269205   0.430645   0.427485]
                          [0.582374   0.445654   0.581855   0.536988]
                          [0.466896   0.480944   0.566734   0.500487]
                          [0.744044   0.516233   0.672581   0.609990]
                          [0.212844   0.516233   0.158468   0.135476]])
          :outputs (array [[0 1 0]
                           [0 1 0]
                           [0 1 0]
                           [0 1 0]
                           [1 0 0]])
          :classes (array [[2]
                           [2]
                           [2]
                           [2]
                           [1]])})

(util/matrix-2d-pretty-print (:inputs foo))