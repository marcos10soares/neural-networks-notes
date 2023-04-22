On-line learning, also known as incremental learning or sequential learning, is a learning strategy where the model is continuously updated with new data as it becomes available, rather than being trained in a batch fashion with all the data at once. This approach is particularly useful when data is generated over time or is too large to fit in memory.

There are different strategies to perform on-line learning in neural networks:

    Adapting only linear weights:
    In this strategy, the non-linear weights (i.e., the weights of the hidden layers) are pre-trained using an offline method (e.g., backpropagation with the available data), and only the linear weights (i.e., the weights of the output layer) are updated during the on-line learning process. This approach assumes that the non-linear features learned during the pre-training phase are general enough and only the final decision-making layer (output layer) needs to be adapted to the new data.

Pros:

    Less computationally expensive compared to updating all weights in the network.
    Faster convergence, as only the output layer is being updated.

Cons:

    The network may not adapt as effectively to the new data, as the non-linear weights are fixed.

    Starting from scratch and determining all weights during on-line learning:
    In this approach, the model starts with random initial weights, and all the weights in the network are updated during the on-line learning process. This can be done using on-line learning algorithms, such as stochastic gradient descent (SGD), where the weights are updated for each incoming data point (or mini-batch) instead of updating them based on the entire dataset.

Pros:

    More flexible, as the entire network adapts to the new data.
    Can better capture complex and changing patterns in the data.

Cons:

    Computationally more expensive, as all weights in the network need to be updated.
    May require more time to converge.

In summary, on-line learning methods offer a way to update models continuously as new data becomes available. The choice between adapting only the linear weights or determining all weights during the on-line learning process depends on the specific problem, available resources, and how quickly the underlying data patterns are expected to change.