(ns cloann.gelnn.nn-instructions
  (:use [clojush pushstate globals]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Utility Functions

(defn insert
  "https://groups.google.com/forum/#!msg/clojure/zjZDSziZVQk/58xCUSZYYPwJ"
  [vec pos item] 
  (apply merge (subvec vec 0 pos) item (subvec vec pos)))

(defn insert-hidden-layer
  "Returns network-info with new hidden layer."
  [network-info num-nodes]
  (let [layers (:layers network-info)
        hidden-layers (dissoc (dissoc layers
                                      :I)
                              :O)
        next-hidden-id (symbol (str ":H" 
                                    (inc (count hidden-layers))))]
    [(assoc-in network-info
               [:layers next-hidden-id]
               {:num-nodes num-nodes})
     next-hidden-id]))

(defn new-layer-connection
  "Returns network-info with a new layer connection."
  [network-info [from-id to-id]]
  (let [layer-conns (:layer-connections network-info)
        output-connections (filter #(= :O (nth % 1)) 
                                   layer-conns)
        non-output-connections (remove  #(= :O (nth % 1))
                                        layer-conns)
        new-layer-conns (concat (conj (vec non-output-connections)
                                      [from-id to-id])
                                output-connections)]
    (if (some #(= [from-id to-id] %) layer-conns)
      network-info
      (assoc network-info :layer-connections new-layer-conns))))

(defn remove-layer-connection
  "Returns network-info with layer connection removed."
  [network-info [from-id to-id]]
  (let [layer-conns (:layer-connections network-info)
        new-layer-conns (remove #(= [from-id to-id] %)
                                layer-conns)]
    (assoc network-info :layer-connections new-layer-conns)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Instructions


; Connects two layers based on their index in the list of layers
(define-registered
  nn_connect_layers
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (rest (:integer state))))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          to-index (top-item :integer state)
          from-index (top-item :integer (pop-item :integer state))
          from-id (nth (keys (:layers nn-info))
                       (mod from-index
                            (count (keys (:layers nn-info)))))
          to-id (nth (keys (:layers nn-info))
                     (mod to-index
                          (count (keys (:layers nn-info)))))
          new-nn-info (new-layer-connection nn-info [from-id to-id])]
      (->> 
        (pop-item :integer state)
        (pop-item :integer)
        (pop-item :auxilary)
        (push-item new-nn-info :auxilary)))
    state)))

(define-registered
  nn_bud
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-bud (nth (:layer-connections nn-info)
                           (mod (stack-ref :integer 0 state)
                                (count (:layer-connections nn-info))))
          num-nodes (:num-nodes (get (:layers nn-info)
                                     (second edge-to-bud)))
          
          temp (insert-hidden-layer nn-info 
                                    num-nodes)
          new-nn-info (first temp)
          new-hidden-id (second temp)
          
          new-new-nn-info (new-layer-connection new-nn-info [(second edge-to-bud)
                                                             new-hidden-id])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item new-new-nn-info :auxilary)))
    state)))

(define-registered
  nn_split
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-split (nth (:layer-connections nn-info)
                             (mod (stack-ref :integer 0 state)
                                  (count (:layer-connections nn-info))))
          num-nodes(:num-nodes (get (:layers nn-info)
                                    (first edge-to-split)))
          
          temp (insert-hidden-layer (remove-layer-connection nn-info 
                                                             edge-to-split)
                                    num-nodes)
          nn-info (first temp)
          new-hidden-id (second temp)
          
          nn-info (new-layer-connection nn-info [(first edge-to-split) new-hidden-id])
          nn-info (new-layer-connection nn-info [new-hidden-id (second edge-to-split)])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item nn-info :auxilary)))
    state)))

(define-registered
  nn_loop
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-loop (nth (:layer-connections nn-info)
                            (mod (stack-ref :integer 0 state)
                                 (count (:layer-connections nn-info)))) 
          nn-info (new-layer-connection nn-info [(second edge-to-loop)
                                                 (first edge-to-loop)])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item nn-info :auxilary)))
    state)))

(define-registered
  nn_reverse
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-reverse (nth (:layer-connections nn-info)
                               (mod (stack-ref :integer 0 state)
                                    (count (:layer-connections nn-info))))]
      (if (or (= (first edge-to-reverse)
                 :I)
              (= (second edge-to-reverse)
                 :O))
        state
        (let [nn-without-edge (remove-layer-connection nn-info edge-to-reverse)
              nn-info (new-layer-connection nn-without-edge [(second edge-to-reverse)
                                                             (first edge-to-reverse)])]
          (->> 
            (pop-item :integer state)
            (pop-item :auxilary)
            (push-item nn-info :auxilary)))))
    state)))

(define-registered
  nn_set_num_nodes_layer
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (rest (:integer  state))))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          layer-ids (keys (:layers nn-info))
          layer-id (nth layer-ids
                        (mod (stack-ref :integer 0 state)
                             (count layer-ids)))
          new-num-nodes (stack-ref :integer 1 state)]
      (if (or (= layer-id :I)
              (= layer-id :O))
        state
        (let [new-nn-info (assoc-in nn-info
                                    [:layers layer-id :num-nodes]
                                    new-num-nodes)]
          (->> 
            (pop-item :integer state)
            (pop-item :integer)
            (pop-item :auxilary)
            (push-item new-nn-info :auxilary)))))
    state)))