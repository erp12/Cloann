(ns cloann.examples.iris
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]
            [cloann.transfer-functions :as tran-funcs]))

;; Matrix of all the data from the csv file
(def training-data-matrix
  (dIO/csv->matrix "data/iris_training.csv" false))
(def testing-data-matrix
  (dIO/csv->matrix "data/iris_testing.csv" false))
(def validation-data-matrix
  (dIO/csv->matrix "data/iris_validation.csv" false))

;(def topology-encoding 
;  {:layers  {:I {:num-nodes 4}
;             :H1 {:num-nodes 8}
;             :H2 {:num-nodes 6}
;             :O {:num-nodes 3}}
;   :layer-connections [[:I  :H1]
;                       [:H1 :H2]
;                       [:H2 :O]]})

(def topology-encoding
  {:layers {:I {:num-nodes 4}
            :O {:num-nodes 3}
            :H1 {:num-nodes 4}
            :H2 {:num-nodes 4}}
   :layer-connections [[:I :H1] [:H1 :H2] [:H2 :H1] [:I :O] [:H1 :O]]}
)

(def iris-nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices training-data-matrix
                                                    testing-data-matrix
                                                    validation-data-matrix
                                                    [0 1 2 3] ; Input indexes
                                                    [4 5 6]) ; Output indexes
   :max-epochs 300
   :max-weight-initial 0.2
   :learning-rate 0.1
   :validation-stop-threshold 0.02})

(cloann/run-cloann iris-nn-params topology-encoding true)


;;;;;;;;;;;;;;;;
; UNIT TESTING ;
;;;;;;;;;;;;;;;;


(comment
(swap! cloann/nn-params #(merge % iris-nn-params))

(swap! cloann/nn-params (fn [i] (assoc i
                                       :topology-encoding
                                       (util/remove-useless-topology-from-encoding (:topology-encoding i)))))
(swap! cloann/nn-params (fn [i] (assoc-in i
                                          [:topology-encoding :layer-connections]
                                          (util/sort-layer-connections (:layer-connections (:topology-encoding i))))))
(swap! cloann/nn-params (fn [i] (assoc-in i
                                          [:topology-encoding :layer-order-in-weight-matrix]
                                          (vec (keys (:layers (:topology-encoding i)))))))

(println  (:topology-encoding @cloann/nn-params))
(println)
)

(comment
(def uwm (cloann/generate-uninitialized-weight-matrix (:layers (:topology-encoding @cloann/nn-params))
                                                      (:layer-connections (:topology-encoding @cloann/nn-params))
                                                      (:layer-order-in-weight-matrix (:topology-encoding @cloann/nn-params))))
;(util/matrix-2d-pretty-print uwm)
;(println)

(def wm (cloann/initialize-weights uwm
                                   0))

(util/matrix-2d-pretty-print wm)
(println)
)

(comment
(def blah (util/replace-layer-connection-weights-in-matrix wm
                                                           (:layers (:topology-encoding @cloann/nn-params))
                                                           (:layer-order-in-weight-matrix (:topology-encoding @cloann/nn-params))
                                                           [:H1 :H1]
                                                           [[:AA :AA :AA :AA]
                                                            [:AA :AA :AA :AA]
                                                            [:AA :AA :AA :AA]
                                                            [:AA :AA :AA :AA]]))

(util/matrix-2d-pretty-print blah)
)

;(def sub-wm (util/get-connection-matrix-by-id-with-bias wm
;                                                        (:layers (:topology-encoding @cloann/nn-params)) 
;                                                        [:I :O :H1 :H2] 
;                                                        [:I :H1]))
;(util/matrix-2d-pretty-print sub-wm)
;(println)

;(def ff (cloann/feed-forward (first (:inputs (:training-set (:data-sets @cloann/nn-params))))
;                             wm))

;ff
a
;(def bp (cloann/backpropagation (first (:inputs (:training-set (:data-sets @cloann/nn-params))))
;                                (first (:outputs (:training-set (:data-sets @cloann/nn-params))))
;                                wm
;                                0.1))
;(util/matrix-2d-pretty-print bp)
