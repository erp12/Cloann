(ns cloann.activation-functions
  (:use [cloann util])
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))

(defn hyperbolic-tangent
  [x]
  (/ (+ (Math/tanh x)
        1)
     2))

(defn hyperbolic-tangent-derivative 
  [x]
  (/ (- 1
        (* (Math/tanh x)
           (Math/tanh x)))
     2))

(defn sigmoid
  [x]
  (/ 1
     (+ 1
        (Math/pow Math/E
                  x))))

(defn sigmoid-derivative
  [x]
  (* (sigmoid x)
     (- 1
        (sigmoid x))))

(defn ReLU
  [x]
  (max 0 x))

(defn ReLU-derivative
  [x]
  -1)

(defn softplus
  "Smoothed ReLU"
  [x]
  (nat-log (+ 1
              (Math/pow Math/E x))))

(defn softplus-derivative
  [x]
  (/ 1
     (+ 1
        (Math/pow Math/E x))))