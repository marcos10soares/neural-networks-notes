A learning criterion, also known as the loss function or objective function, is a measure used to quantify the difference between the predicted output and the actual output (or target) in a machine learning model. The goal of training a model is to minimize this difference by adjusting the model's parameters (weights and biases). The learning criterion helps guide the optimization process by providing feedback on how well the model is performing.

There are various learning criteria, and the choice depends on the problem you are trying to solve. Some common learning criteria include mean squared error (MSE), cross-entropy loss, and hinge loss.

Now, let's discuss linear and nonlinear weights:

    Linear weights: In the context of neural networks, linear weights refer to the model parameters when the activation functions in the network are linear. Linear functions have the form y = ax + b, where 'a' and 'b' are constants. When a neural network uses linear activation functions, the overall model is also linear. In this case, the learning criterion is a quadratic function of the weights, and optimization can be performed using closed-form solutions, such as the normal equation in linear regression.

    Nonlinear weights: Nonlinear weights refer to the model parameters when the activation functions in the network are nonlinear, such as sigmoid, ReLU, or tanh functions. When a neural network uses nonlinear activation functions, the overall model becomes nonlinear, and the learning criterion is a non-quadratic function of the weights. Optimization in this case becomes more challenging, and iterative optimization algorithms like gradient descent or quasi-Newton methods are usually employed to minimize the learning criterion.

In summary, a learning criterion is a measure that quantifies the performance of a machine learning model. Linear and nonlinear weights refer to the parameters of a model depending on whether the activation functions are linear or nonlinear, which influences the shape of the learning criterion and the optimization techniques employed to minimize it.


In the context of learning criteria, QN, LM, and BP are not directly related to the learning criterion itself, but rather to the optimization algorithms used to minimize the learning criterion in a neural network. These abbreviations stand for:

    - QN: Quasi-Newton - Quasi-Newton methods are a class of optimization algorithms that approximate the second-order (Hessian) information of the learning criterion using first-order (gradient) information. These methods aim to achieve faster convergence than gradient-based methods like Gradient Descent. Some popular Quasi-Newton methods include Broyden-Fletcher-Goldfarb-Shanno (BFGS) and Limited-memory BFGS (L-BFGS).

    - LM: Levenberg-Marquardt - The Levenberg-Marquardt algorithm is an optimization method that combines the ideas of Gradient Descent and the Gauss-Newton algorithm, which is a second-order optimization method for nonlinear least squares problems. It is particularly well-suited for optimizing the learning criterion in feedforward neural networks with a small to moderate number of parameters.

    - BP: Backpropagation - Backpropagation is an algorithm used to train neural networks by minimizing the learning criterion. It is a supervised learning method that computes the gradient of the learning criterion with respect to each weight by applying the chain rule of calculus. The gradients are then used to update the weights using an optimization algorithm, such as Gradient Descent or its variants (e.g., Stochastic Gradient Descent, Adam, RMSProp).

These three optimization methods are used in the context of neural networks to minimize the learning criterion, guiding the model to learn the best set of weights to make accurate predictions. They are not learning criteria themselves but are essential components of the learning process in neural networks.