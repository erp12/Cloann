(ns cloann.examples.student
  (:require [cloann.core :as cloann]
            [cloann.dataIO :as dIO]
            [cloann.util :as util]
            [cloann.transfer-functions :as tran-funcs]))

;; Matrix of all the data from the csv file
(def training-data-matrix
  (dIO/csv->matrix "data/norm_student_alc_training.csv" true))
(def testing-data-matrix
  (dIO/csv->matrix "data/norm_student_alc_testing.csv" true))
(def validation-data-matrix
  (dIO/csv->matrix "data/norm_student_alc_validation.csv" true))

(def topology-encoding
  {:layers {:I {:num-nodes 24}
            :O {:num-nodes 3}
;            :H1 {:num-nodes 3}
;            :H3 {:num-nodes 3}
            }
   :layer-connections [[:I :O]]}
)

(def student-nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices training-data-matrix
                                                    testing-data-matrix
                                                    validation-data-matrix
                                                    (vec (range 24)) ; Input indexes
                                                    [24 25 26]) ; Output indexes
   :topology-encoding topology-encoding
   :max-epochs 500
   :max-weight-initial 0.2
   :learning-rate 0.05
   :validation-stop-threshold 0.02})

(cloann/run-cloann student-nn-params true)


;;;;;;;;;;;;;;;;
; UNIT TESTING ;
;;;;;;;;;;;;;;;;


(comment
(swap! cloann/nn-params #(merge % student-nn-params))

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

;(def sub-wm (util/get-connection-matrix-by-id-with-bias wm
;                                                        (:layers (:topology-encoding @cloann/nn-params)) 
;                                                        [:I :O :H1 :H2] 
;                                                        [:I :H1]))
;(util/matrix-2d-pretty-print sub-wm)
;(println)

;(def ff (cloann/feed-forward (first (:inputs (:training-set (:data-sets @cloann/nn-params))))
;                             wm))

;ff

;(def bp (cloann/backpropagation (first (:inputs (:training-set (:data-sets @cloann/nn-params))))
;                                (first (:outputs (:training-set (:data-sets @cloann/nn-params))))
;                                wm
;                                0.1))
;(util/matrix-2d-pretty-print bp)
