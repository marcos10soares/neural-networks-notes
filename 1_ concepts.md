# Concepts
- neurons
- ANNs
- Activation functions
    - Sign, or Threshold Function • Piecewise-linear Function
    - Linear Function
    - Sigmoid Function
    - Hyperbolic Tangent
    - Gaussian Function


# Characteristics of Neural Networks
Neural network models can be somehow characterized by
- the models employed for the individual neurons,
- the way that these neurons are interconnected between themselves,
- and by the learning mechanism(s) employed.

## Interconnecting neurons
- As already mentioned, biological neural networks are densely interconnected. This also happens with artificial neural networks.
- According to the flow of the signals within an ANN, we can divide the architectures into:
    - feedforward networks, if the signals flow just from input to output, 
    - recurrent networks, if loops are allowed.
- According to the existence of hidden neurons, we can have: 
    - multilayer NNs, if there are hidden neurons,
    - singlelayer NN, if no hidden neurons (and layers) exist.
- Finally, the NNs are:
    - fully connected, if every neuron in one layer is connected with the layer immediately above
    - partially connected, if not.

## Learning can be divided into three classes
- supervised learning: this learning scheme assumes that the network is used as an input-output system. Because of that, for each input pattern there is a desired output, or target, and a cost function is used to update the parameters;
    - Gradient descent learning
    - forced Hebbian or correlative learning
- reinforcement learning: In contrast with supervised learning, however, this cost function is only given to the network from time to time. The network does not receive a teaching signal at every training pattern but only a score that tells it how it performed over a training sequence;
- unsupervised learning: in this case there is no teaching signal from the environment.
    - Hebbian learning
    - competitive learning
