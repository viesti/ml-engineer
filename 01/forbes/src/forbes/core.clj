(ns forbes.core
  (:require [kixi.stats.core :as kixi]
            [kixi.stats.math :refer [log]]
            [net.cgrand.xforms :as x]
            [jutsu.core :as j]
            [clojure.java.io :as io]
            [clojure-csv.core :as csv]
            [semantic-csv.core :as sc]
            [semantic-csv.transducers :as sct]))

(set! *print-length* nil)

(defn parse-data
  "Turns Forbes2000.csv file into a sequence of maps"
  [path]
  (with-open [in (io/reader path)]
    (->> (csv/parse-csv in)
         (map #(drop 1 %))
         (sc/mappify)
         (sc/cast-with {:profits #(try (Double/parseDouble %) (catch Throwable t nil))
                        :assets #(Double/parseDouble %)
                        :rank #(Long/parseLong %)
                        :sales #(Double/parseDouble %)
                        :marketvalue #(Double/parseDouble %)})
         doall)))

(defn x-parse-data
  "Same as parse-data but with transducers (without intermediate seqs)"
  [path]
  (with-open [in (io/reader path)]
    (into []
          (comp (map #(drop 1 %))
                (sct/mappify)
                (sct/cast-with {:profits #(try (Double/parseDouble %) (catch Throwable t nil))
                                :assets #(Double/parseDouble %)
                                :rank #(Long/parseLong %)
                                :sales #(Double/parseDouble %)
                                :marketvalue #(Double/parseDouble %)}))
          (csv/parse-csv in))))

(def data (x-parse-data "Forbes2000.csv"))

(j/start-jutsu!)

;; https://skillsmatter.com/skillscasts/9050-clojure-for-machine-learning

;; Clojure for Data Science Book, section: Visualizing data

(j/graph! "marketvalue histogram"
          [{:x (->> data (map :marketvalue))
            :type "histogram"
            :name "marketvalue"}])

(j/graph! "sales histogram"
          [{:x (->> data (map :sales))
            :type "histogram"
            :name "marketvalue"}])

(j/graph! "assets histogram"
          [{:x (->> data (map :assets))
            :type "histogram"
            :name "marketvalue"}])

(j/graph! "sales log histogram"
          [{:x (map (comp log :sales) data)
            :type "histogram"
            :name "sales"}])

(j/graph! "marketvalue log histogram"
          [{:x (map (comp log :marketvalue) data)
            :type "histogram"
            :name "marketvalue"}])

(j/graph! "marketvalue by sales log scatter"
          [{:y (map (comp #(Math/log %) :marketvalue) data)
            :x (map (comp #(Math/log %) :sales) data)
            :type "scatter"
            :mode "markers"
            :name "marketvalue"}])

(def model (transduce (comp (map #(update % :marketvalue log))
                            (map #(update % :sales log)))
                      (kixi/simple-linear-regression :sales :marketvalue)
                      data))

(defn y-at-x [[a b] x]
  (+ (* x a) b))

(def data-log-scale
  (sequence (comp (map #(update % :marketvalue log))
                  (map #(update % :sales log))
                  (map #(update % :assets log)))
            data))

(defn bounds [data]
  {:min-y (reduce min (map :marketvalue data))
   :max-y (reduce max (map :marketvalue data))
   :min-x (reduce min (map :sales data))
   :max-x (reduce max (map :sales data))})


;; Single pass
(defn bounds-marketvalue-sales [data]
  (into {} (x/transjuxt {:min-y (comp (map :marketvalue) x/min)
                         :max-y (comp (map :marketvalue) x/max)
                         :min-x (comp (map :sales) x/min)
                         :max-x (comp (map :sales) x/max)})
        data))

(j/graph! "marketvalue by sales log scatter with model"
          [{:y (map :marketvalue data-log-scale)
            :x (map :sales data-log-scale)
            :type "scatter"
            :mode "markers"
            :name "marketvalue"}
           (let [{:keys [min-x max-x min-y max-y]} (bounds-marketvalue-sales data-log-scale)]
             {:y [(y-at-x model min-x) (y-at-x model max-x)]
              :x [min-x max-x]
              :type "scatter"
              :mode "lines+markers"
              :name "model"})])

(def sales-marketvalue-correlation-by-country
  (->> (for [country (->> data (map :country) set)]
         [country (transduce (filter #(-> % :country (= country)))
                             (kixi/correlation :sales :marketvalue)
                             data)])
             (sort-by second #(compare %2 %1))))

(j/graph! "Sales marketvalue correlation by country"
          [{:x (map first sales-marketvalue-correlation-by-country)
            :y (map second sales-marketvalue-correlation-by-country)
            :type "bar"}])

(def sales-marketvalue-correlation-by-category
  (->> (for [category (->> data (map :category) set)]
         [category (transduce (filter #(-> % :category (= category)))
                              (kixi/correlation :sales :marketvalue)
                              data)])
       (sort-by second #(compare %2 %1))))

(def category-10-best-correlation
  (->> sales-marketvalue-correlation-by-category
       (take 10)
       (map first)
       set))

(j/graph! "marketvalue by sales log from 10 best correlation by category"
          [{:y (sequence (comp (filter #(category-10-best-correlation (:category %)))
                               (map :marketvalue)) data-log-scale)
            :x (sequence (comp (filter #(category-10-best-correlation (:category %)))
                               (map :sales)) data-log-scale)
            :type "scatter"
            :mode "markers"
            :name "marketvalue by sales"}])

(for [[category correlation] (->> sales-marketvalue-correlation-by-category
                                  (take 5))]
  (j/graph! (str category ": marketvalue by sales ")
            (let [pred #(= category (:category %1))
                  category-data (filter pred data-log-scale)
                  category-model (transduce identity
                                            (kixi/simple-linear-regression :sales :marketvalue)
                                            category-data)
                  {:keys [min-x max-x min-y max-y]} (bounds-marketvalue-sales category-data)]
              [{:y (map :marketvalue category-data)
                :x (map :sales category-data)
                :type "scatter"
                :mode "markers"
                :name (format "%s, %.02f" category correlation)}
               {:y [(y-at-x category-model min-x) (y-at-x category-model max-x)]
                :x [min-x max-x]
                :type "scatter"
                :mode "lines+markers"
                :name "model"}])))

(def category-5-best-correlation
  (->> sales-marketvalue-correlation-by-category
       (take 5)
       (map first)
       set))

#_(defn normal-equation [x y]
  (let [xtx  (i/mmult (i/trans x) x)
        xtxi (i/solve xtx)
        xty  (i/mmult (i/trans x) y)]
    (i/mmult xtxi xty)))
