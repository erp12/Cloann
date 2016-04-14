(defproject cloann "0.0.1-SNAPSHOT"
  :description "FIXME: write description"
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [net.mikera/vectorz-clj "0.35.0"]
                 [org.clojure/data.csv "0.1.3"]
                 [incanter "1.5.6"]
                 [org.clojure/math.combinatorics "0.1.1"]
                 [clojush "2.0.63"]]
  :javac-options ["-target" "1.6" "-source" "1.6" "-Xlint:-options"]
  :jvm-opts ["-XX:-OmitStackTraceInFastThrow"]
  :aot [cloann.core]
  :main cloann.core)
