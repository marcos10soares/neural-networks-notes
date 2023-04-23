In the context of pattern recognition, statistical classifiers aim to assign an input pattern to one of several predefined classes based on the probabilistic description of the classes. In our example, we have two classes, "health" and "sickness," and we want to classify a patient as healthy or sick based on their temperature, which is considered a random variable.

Let's denote the classes as $C_1$ (healthy) and $C_2$ (sick), and let the random variable $T$ represent the patient's temperature. The goal is to estimate the probability that a patient belongs to a particular class given their temperature, i.e., $P(C_1|T)$ and $P(C_2|T)$.

## Bayes's Rule

Bayes's Rule relates the conditional probabilities of the classes and the random variable. It states that:

$$
P(C_i|T) = P(T|C_i) * P(C_i) / P(T)
$$

Where:

- $P(C_i|T)$ is the posterior probability, the probability that a patient belongs to class $C_i$ given their temperature $T$.
- $P(T|C_i)$ is the likelihood, the probability of observing temperature T given that the patient belongs to class $C_i$.
- $P(C_i)$ is the prior probability, the probability that a patient belongs to class $C_i$ before observing their temperature.
- $P(T)$ is the evidence, the probability of observing temperature $T$.

To build a statistical classifier using Bayes's Rule, we need to estimate the likelihoods and priors. In our example, we can estimate these probabilities from a dataset containing the temperatures of healthy and sick patients.

1. Estimate the likelihoods:
    - $P(T|C_1)$: This could be the probability density function of temperature for healthy patients, which could be modeled as a Gaussian distribution with a certain mean and variance.
    - $P(T|C_2)$: Similarly, this could be the probability density function of temperature for sick patients, modeled as another Gaussian distribution with a different mean and variance.

2. Estimate the priors:
    - $P(C_1)$: The probability that a patient is healthy. This can be calculated as the proportion of healthy patients in the dataset.
    - $P(C_2)$: The probability that a patient is sick. This can be calculated as the proportion of sick patients in the dataset.

Once we have the likelihoods and priors, we can use Bayes's Rule to classify a new patient based on their temperature:

3. Calculate the posterior probabilities for each class:
    - $P(C_1|T)$: Probability that the patient is healthy given their temperature.
    - $P(C_2|T)$: Probability that the patient is sick given their temperature.

4. Assign the patient to the class with the highest posterior probability.

In this way, Bayes's Rule forms the basis for the statistical formulation of classifiers in pattern recognition. By estimating the likelihoods and priors from data, we can build a probabilistic model to classify new patterns based on their features, in this case, the patient's temperature.

## Statistical decision theory 

Statistical decision theory is a framework that combines probability theory, statistics, and decision making to make optimal choices in uncertain situations. In the context of pattern recognition, it can be used to construct the optimal classifier by minimizing the misclassification risk under the given constraints.

In statistical decision theory, we have a set of possible decisions (classifications) and a set of outcomes (true classes). We assign a cost to each combination of decisions and outcomes, which is represented by a loss function. The goal is to find a decision rule (classifier) that minimizes the expected loss (misclassification risk).

Bayesian decision theory is a specific application of statistical decision theory based on Bayes's rule. Bayes's rule states that the probability of a hypothesis (class) given observed data is proportional to the product of the prior probability of the hypothesis and the likelihood of the data given the hypothesis:

$$
P(H|D) = (P(D|H) * P(H)) / P(D)
$$

In pattern recognition, we can use Bayes's rule to calculate the posterior probabilities of each class given the observed data (features). The optimal classifier based on Bayesian decision theory is the one that assigns an observation to the class with the highest posterior probability. This is known as the Bayes optimal classifier or the maximum a posteriori (MAP) decision rule.

The Bayesian threshold is a specific case of the Bayes optimal classifier when we have only two classes. The threshold value is chosen such that the ratio of posterior probabilities of the two classes is equal to the ratio of the costs associated with misclassification. In other words, we set a threshold T such that:

$$
P(H1|D) / P(H2|D) = C(H1, H2) / C(H2, H1)
$$

Where $C(H1, H2)$ is the cost of classifying an observation as $H1$ when it is actually $H2$, and $C(H2, H1)$ is the cost of classifying an observation as $H2$ when it is actually $H1$. When the ratio of posterior probabilities is greater than the threshold, we classify the observation as $H1$; otherwise, we classify it as $H2$.

In summary, statistical decision theory, and specifically Bayesian decision theory, can be used to construct the optimal classifier by minimizing the expected loss (misclassification risk). In the two-class case, the Bayesian threshold is a criterion that helps us decide which class to assign an observation based on the ratio of posterior probabilities and the costs associated with misclassification.

## Discriminant functions and Optimal classifiers

Discriminant functions are mathematical functions that map input feature vectors to their corresponding class labels. They are used to separate different classes in a pattern recognition problem. Discriminant functions assign a score to each class for a given input, and the class with the highest score is chosen as the predicted class. In other words, discriminant functions are used to make a decision about which class an input belongs to.

Optimal classifiers are classifiers that minimize the probability of misclassification or minimize the expected cost of misclassification. These classifiers aim to achieve the best possible classification performance on the available data. Often, the optimality of a classifier is based on certain assumptions about the data, such as the underlying probability distribution or the cost function associated with misclassification.

In the context of statistical pattern recognition, optimal classifiers can be designed by utilizing the properties of the class-conditional probability density functions (pdfs). One such classifier is the Bayes classifier, which is considered optimal when the class-conditional pdfs and the prior probabilities of the classes are known. The Bayes classifier minimizes the probability of error by choosing the class with the highest posterior probability for a given input.

To summarize, discriminant functions are used to separate classes in a pattern recognition problem, and optimal classifiers are those that minimize the probability of misclassification or the expected cost of misclassification. The design of optimal classifiers typically relies on assumptions about the data, such as the class-conditional pdfs and prior probabilities.