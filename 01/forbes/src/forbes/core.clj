(ns forbes.core
  (:require [kixi.stats.core :as kixi]
            [kixi.stats.math :refer [log]]
            [net.cgrand.xforms :as x]
            [jutsu.core :as j]
            [clojure.java.io :as io]
            [clojure-csv.core :as csv]
            [semantic-csv.core :as sc]
            [semantic-csv.transducers :as sct]
            [clojure.core.matrix.linear :as ml]
            [clojure.core.matrix :as m]))

(set! *print-length* nil)
(m/set-current-implementation :vectorz)

(defn parse-data
  "Turns csv file into vector of maps"
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
(Thread/sleep 4000) ;; wait for jutsu to start

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

(defn linear-model [x y data]
  (transduce identity
             (kixi/simple-linear-regression x y)
             data))

(defn normal-equation
  "core.matrix version of normal-equation from \"Clojure for Data Science\" book:
https://github.com/clojuredatascience/ch3-correlation/blob/master/src/cljds/ch3/stats.clj#L89-L93"
  [x y]
  (let [xtx  (m/mmul (m/transpose x) x)
        xtxi (ml/solve xtx)
        xty  (m/mmul (m/transpose x) y)]
    (m/mmul xtxi xty)))

(defn add-bias [x]
  (if (seqable? x)
    (into [1] x)
    [1 x]))

(defn linear-model-matrix [x y data]
  "Linear model using normal equation, copes with more than one independent variable"
  (let [x (map (comp add-bias x) data)
        y (map y data)]
    (normal-equation x y)))

(defn y-at-x [[offset slope] x]
  (+ (* x slope) offset))

(defn regression-line [model]
  (fn [x]
    (y-at-x model x)))

(def data-log-scale
  (sequence (comp (map #(update % :marketvalue log))
                  (map #(update % :sales log))
                  (map #(update % :assets log)))
            data))

(defn bounds
  "Returns map with :min-x, :max-x, min-y and max-y of given x and y fns"
  [x y data]
  (into {} (x/transjuxt {:min-y (comp (map y) x/min)
                         :max-y (comp (map y) x/max)
                         :min-x (comp (map x) x/min)
                         :max-x (comp (map x) x/max)})
        data))

(j/graph! "marketvalue by sales log scatter with model"
          [{:y (map :marketvalue data-log-scale)
            :x (map :sales data-log-scale)
            :type "scatter"
            :mode "markers"
            :name "marketvalue"}
           (let [model (linear-model :sales :marketvalue data-log-scale)
                 {:keys [min-x max-x min-y max-y]} (bounds :sales :marketvalue data-log-scale)
                 estimate (regression-line model)]
             {:y [(estimate min-x) (estimate max-x)]
              :x [min-x max-x]
              :type "scatter"
              :mode "lines+markers"
              :name "model"})])

(defn make-residual [model]
  (let [estimate (regression-line model)]
    (fn [x y] (- y (estimate x)))))

(defn r-squared
  "kixi.stats version of https://github.com/clojuredatascience/ch3-correlation/blob/master/src/cljds/ch3/stats.clj#L75-L79"
  [model x y data]
  (let [residual (make-residual model)
        r-var (transduce (map #(residual (x %) (y %))) kixi/variance data)
        y-var (transduce (map y) kixi/variance data)]
    (- 1 (/ r-var y-var))))

(j/graph! "residuals"
          (let [model (linear-model :sales :marketvalue data-log-scale)
                residual (make-residual model)]
            [{:y (map #(residual (:sales %) (:marketvalue %)) data-log-scale)
              :x (map :sales data-log-scale)
              :type "scatter"
              :mode "markers"
              :name "residuals"}]))

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

(for [[category correlation] (take 5 sales-marketvalue-correlation-by-category)]
  (j/graph! (str category ": marketvalue by sales ")
            (let [pred #(= category (:category %1))
                  category-data (filter pred data-log-scale)
                  category-model (transduce identity
                                            (kixi/simple-linear-regression :sales :marketvalue)
                                            category-data)
                  {:keys [min-x max-x min-y max-y]} (bounds :sales :marketvalue category-data)
                  r2 (r-squared category-model :sales :marketvalue category-data)]
              [{:y (map :marketvalue category-data)
                :x (map :sales category-data)
                :type "scatter"
                :mode "markers"
                :name (format "%s, cor: %.02f, R^2: %.02f" category correlation r2)}
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

(j/graph! "marketvalue by sales and assets log scatter with model"
          [{:y (map :marketvalue data-log-scale)
            :x (map :sales data-log-scale)
            :type "scatter"
            :mode "markers"
            :name "marketvalue by sales"}
           {:y (map :marketvalue data-log-scale)
            :x (map :assets data-log-scale)
            :type "scatter"
            :mode "markers"
            :name "marketvalue by assets"}
           (let [model (linear-model-matrix :sales :marketvalue data-log-scale)
                 {:keys [min-x max-x min-y max-y]} (bounds :sales :marketvalue data-log-scale)]
             {:y [(y-at-x model min-x) (y-at-x model max-x)]
              :x [min-x max-x]
              :type "scatter"
              :mode "lines+markers"
              :name "sales model"})
           (let [model (linear-model-matrix :assets :marketvalue data-log-scale)
                 {:keys [min-x max-x min-y max-y]} (bounds :assets :marketvalue data-log-scale)]
             {:y [(y-at-x model min-x) (y-at-x model max-x)]
              :x [min-x max-x]
              :type "scatter"
              :mode "lines+markers"
              :name "assets model"})])
