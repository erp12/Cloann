# Cloann
*Artificial Neural Networks for Clojure*

## Parameter
Key | Value | Default
|---|---|---|
:activation-func | Actication function, or transfer function, of the nodes in the network | hyperbolic tangent
:activation-func-derivative | Derivative of the activation function. Used in backpropagation. | hyperbolic tangent derivative
:max-weight-initial | When generating random weights, the possible values for the weight w is -x <= w <= x, where x is the value of this parameter. 
:data-set | Map of :Training, :Validation, and :Testing data. Must be set before building/running the network. More detail below.| nil
:learning-rate | How stronly backpropagation should change each weight per epoch. Often refered to as eta. | 0.02
:max-epochs | Maximum number of epochs spent training the network. Used as a stopping condition for training. | 500
:debug-prints | When set to true, each epoch will print the results of each evaluation of the network, the change in weights, and the current value of the weights. *This needs cleaning up to be more useful.* | false
:validation-stop-threshold | Used as a stopping condition for training. When the validation error gets below this value, the network stops training. | 0.03

### Data set structure

## Network Representation

## Examples

## ToDo
