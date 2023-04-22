Momentum is a technique used in optimization algorithms, such as gradient descent, to speed up convergence and help escape local minimums. It is inspired by the concept of momentum in classical physics, where a moving object with mass acquires momentum, making it harder to stop or change its direction quickly.

In the context of neural networks and optimization, the momentum parameter is introduced to the update rule for the model's weights. Instead of updating the weights solely based on the current gradient, a weighted average of the previous gradients is also considered. This has the effect of "smoothing" the optimization trajectory and preventing rapid changes in direction.

Mathematically, the momentum update rule is as follows:

v_t = beta * v_(t-1) + (1 - beta) * g_t
w_t = w_(t-1) - learning_rate * v_t

Here:

    v_t is the momentum term at time step t
    beta is the momentum coefficient (usually between 0.5 and 0.9)
    g_t is the gradient of the loss function with respect to the weights at time step t
    w_t is the updated weight at time step t
    learning_rate is the learning rate of the optimization algorithm

The momentum parameter helps escape local minimums in the following ways:

    Smoothing the optimization trajectory: The momentum term averages out the gradients over multiple iterations, resulting in a smoother trajectory. This can help the optimizer avoid oscillations and zig-zagging in the parameter space.

    Escaping local minimums and saddle points: The momentum term adds a "memory" of the previous gradients to the optimization process. This can give the optimizer enough "inertia" to overcome shallow local minimums or saddle points, where the gradients are small or close to zero. By pushing the optimizer out of these regions, the momentum parameter can help find better solutions (i.e., lower minimums) in the parameter space.

    Accelerating convergence: The momentum term can speed up convergence by allowing the optimizer to move faster in the directions of consistent gradients, while dampening oscillations in directions of changing gradients. This can lead to faster convergence to the global minimum or a better local minimum.

In summary, using a momentum parameter in optimization algorithms can help escape local minimums, smooth the optimization trajectory, and accelerate convergence, making it a valuable tool for addressing convergence problems in neural network training.


Oscillations in the first few epochs when using momentum can occur due to the nature of the momentum technique itself. The momentum term is designed to help the optimizer maintain a certain direction during the optimization process by taking into account the previous update steps.

In the initial epochs, the gradients can be quite large, and the momentum term accumulates the gradients from the previous steps. As a result, the updates in the first few epochs can overshoot the optimal point, leading to oscillations in the loss curve.

These oscillations typically diminish over time as the optimizer converges to the optimal point. As the gradient magnitudes decrease, the momentum term becomes less influential, and the updates become more controlled. Using a smaller learning rate can help reduce these oscillations, but it might also lead to slower convergence. One common approach is to use learning rate scheduling, which starts with a larger learning rate and reduces it over time. This way, the optimizer can make larger steps in the initial epochs and smaller, more refined steps as it converges.