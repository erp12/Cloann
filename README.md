# Cloann
**Artificial Neural Networks for Clojure**
Cloann is a clojure library that utilizes matrices to represent, train, and run artificial neural networks.

## Parameter
Key | Value | Default
|---|---|---|
:activation-func | Actication function, or transfer function, of the nodes in the network | hyperbolic tangent
:activation-func-derivative | Derivative of the activation function. Used in backpropagation. | hyperbolic tangent derivative
:max-weight-initial | When generating random weights, the possible values for the weight w is -x <= w <= x, where x is the value of this parameter. 
:data-set | Map of :Training, :Validation, and :Testing data. Must be set before building/running the network. More detail below.| nil
:learning-rate | How stronly backpropagation should change each weight per epoch. Often refered to as eta. | 0.02
:max-epochs | Maximum number of epochs spent training the network. Used as a stopping condition for training. | 500
:debug-prints | When set to true, each epoch will print the results of each evaluation of the network, the change in weights, and the current value of the weights. **This needs cleaning up to be more useful.** | false
:validation-stop-threshold | Used as a stopping condition for training. When the validation error gets below this value, the network stops training. | 0.03

### Data set structure
- Data Set
  - input count: Number of inputs (or input nodes) to the network.
  - output count: Number of outputs (or output nodes) to the network.
  - training set:
    - inputs: Matrix where each row is one set of inputs to the network.
    - outputs: Matrix where each row is one set of outputs to the network, corrisponding to the same row in :inputs.
    - classes: Column vector holding the index of the largest number in each row of outputs. (Ex: Output:'[0 0 1 0]' -> Class:'[2]')
  - validation set:
    - inputs: Matrix where each row is one set of inputs to the network.
    - outputs: Matrix where each row is one set of outputs to the network, corrisponding to the same row in :inputs.
    - classes: Column vector holding the index of the largest number in each row of outputs. (Ex: Output:'[0 0 1 0]' -> Class:'[2]')
  - testing set:
    - inputs: Matrix where each row is one set of inputs to the network.
    - outputs: Matrix where each row is one set of outputs to the network, corrisponding to the same row in :inputs.
    - classes: Column vector holding the index of the largest number in each row of outputs. (Ex: Output:'[0 0 1 0]' -> Class:'[2]')

## Network Representation
**Everything described in this section is still under development**


## Usage / Examples
```clojure
(ns cloann.examples.iris
  (:require [cloann.core :as cloann]                      ; Use cloann.core
            [cloann.dataIO :as dIO]                       ; Use cloann.dataIO, which contains csv and data-set utility functions.
            [cloann.util :as util]                        ; cloann.util contains many utility functions.
            [cloann.activation-functions :as act-funcs])) ; Use cloann.activation-functions, which contains many standard activation functions and their derivatives.

;; Create a matrix out of each csv file.
(def training-data-matrix
  (dIO/csv->matrix "data/iris_training.csv" false))
(def testing-data-matrix
  (dIO/csv->matrix "data/iris_testing.csv" false))
(def validation-data-matrix
  (dIO/csv->matrix "data/iris_validation.csv" false))

(def nn-params
  {:data-sets (dIO/create-data-sets-from-3-matrices training-data-matrix    ; Creates a single data-set out of all 3 sub-sets
                                                    testing-data-matrix     ; and an indication of which columns of the
                                                    validation-data-matrix  ; matrices are inputs and outputs.
                                                    [0 1 2 3] ; Column indexes of inputs.
                                                    [4 5 6])  ; Column indexes of outputs.
   :max-epochs 500                    ; Max training epochs
   :max-weight-intial 0.5             ; Each initial random weight w can be -0.5 < w < 0.5
   :learning-rate 0.02                ; Learning rate for backpropagation.
   :validation-stop-threshold 0.03})  ; Stop training before :max-epochs if error on validation data sub-set drops below this value.

(cloann/run-cloann nn-params)         ; Run cloann with the specified netowrk parameters. Check nn-reporting folder for charts of progress.
```
To use Cloann as a classifier, you will first need 3 `csv` files. One with training data, one with validation data, and one with testing data. These `csv` files can easily be turned into matrices using `cloann.dataIO` utility functions.

Next, you will need to set a few of parameters, detailed above. These parameters should be defined in a map, like in the example.

Finally, you will need to compile your 3 matrices into 1 data-set (using more `cloann.dataIO` utility functions) and set the parameter `:data-set` to your data-set.

Then simply call `(cloann/run-cloann nn-params)` and check the `/nn-reporting/` directory for some plots of the networks error(s) at each epoch. Here is an example of one of these plots for the iris example.

![Iris Error While Training](http://i.imgur.com/cXxCA0l.png)

## ToDo
1. Finish multi-layer network representation.
2. Hook up network representation with Clojush GP for network evolution.
3. Improve performance through concurency.
