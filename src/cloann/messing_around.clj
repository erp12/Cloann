(ns cloann.messing_around
  (:gen-class)
  (:require [cloann.util :as util]
            [cloann.activation-functions :as act-funcs]
            [cloann.dataIO :as dIO]
            [incanter.core :as incntr]
            [incanter.charts :as charts])
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
                   :num-outputs 4}
              :O {:num-inputs 4
                  :num-outputs 3}})
(def layer-conns [[:I :H1][:H1 :H2][:H2 :H1][:H2 :O]])

(defn create-network-matrix
  [layers layer-conns]
  (let []))

(create-network-matrix layers layer-conns)