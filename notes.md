## Supervised ML: Regression and Classification

Linear Regression

- Cost function
- Learning rate

Multiple linear regression

- Vectorization

Gradient Descent (sklearn.linear_model.SGDRegressor)

- Feature scaling
    - Feature scaling
    - Mean normalization
    - Z-score normalization (sklearn.preprocessing.StandardScaler)
- Feature engineering
- Polynomial regression

Classification with logistic regression

- Sigmoid function
    - Decision boundary
- Logistic regression applies sigmoid to the linear regression model
- Loss/ Cost function
- Gradient descent in logistic regression
- Overfitting
    - Underfit: high bias
    - Overfit: high variance
    - Regularization to address overfitting
        - Cost function
        - Regularised linear/ logistic regression

## Advanced Algorithms

Neutral networks

- Sigmoid activation function, NN layers
- Forward propagation (making predictions)
- Tensorflow, Python (NumPy) implementation for NN
- Vectorization — matrix multiplication, more efficient, clean code

Model training

1. specific how to compute output given input x and parameters w, b
    1. z = np.dot(w,x) + b
    2. f_x = 1/(1+np.exp(-z))
    3. model = Sequential([ Dense(units=25, activation=’sigmoid’), Dense(…), Dense(…)])
2. Specific loss (single) and cost (average) functions
    1. loss = -y * np.log(f_x) - (1-y) * np.log(1-f_x) [logistic loss, binary cross entropy, compare prediction VS loss]
    2. model.compile(loss=BinaryCrossentropy(from_logits=True)), classification
    3. model.compile(loss=MeanSquaredError()), regression
3. Train on data to minimise J(w vector, b), cost function
    1. w = w - alpha * dj_dw
    2. b = b - alpha * dj_db
    3. model.fit(X,y,epochs=100), epochs=no. of steps in gradient descent → backpropagation

Activation functions

- linear activation, output: positive/ negative values
- sigmoid, output: 0/ 1 binary classification
- ReLU rectified linear unit, output: non-negative values, faster than sigmoid
- ReLU for all hidden layers, output layers depend on condition

Multiclass classification (not multilabel classification, don’t be confused)

- Softmax regression
    - last layer: linear activation, model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
    - last layer: softmax, model.compile(loss=SparseCategoricalCrossentropy())

Adam algorithm

- automatically adjust learning rate

### Diagnostics

Bias & Variance

- High bias (underfit)
    - Jtrain high, Jtrain **≈** Jcv (cost)
- High variance (overfit)
    - Jcv >> Jtrain, Jtrain may be low
- High bias & high variance
    - Jtrain will be high, Jcv >> Jtrain
- Regularization parameter λ, lowest cost function
- as λ increases, cost function increases
- big difference between
    - baseline performance & training error Jtrain : high bias
    - training error & cross validation error Jcv : high variance

Learning curves

- as training set size increases, Jcv decreases, Jtrain increases

Fixing high variance

- get more training examples
- try smaller sets of features
- try increasing λ

Fixing high bias

- try getting additional features
- try adding polynomial features
- try decreasing λ

NN:

- does not do well on training set → bigger network
- does not do well on CV set → more data

Confusion Matrix (Predicted : Actual)

- True positive 1:1
- True negative 0:0
- False Positive 1:0
- False Negative 0:1
- Precision = True positive / (True pos + False pos)
- Recall = True positive / (True pos + False neg)
- F1 score = 2 * ((P*R) / (P+R))

Decision Tree

1. How to choose what feature to split on at each decision node?
    1. Maximise purity, minimise impurity
        1. Entropy as a measure of impurity
        2. split evenly 50/50: most impure
        3. 0/100 or 100/0: most pure
        4. p0 = 1 - p1
        5. Entropy H(p1) = -p1 log(p1) - (1-p1) log(1-p1) = -p1 log(p1) - p0 log(p0)
        6. Information Gain = H(p1root) - (wleft * H(p1left) + wright * H(p1right))
2. When do you stop splitting?
    1. When a node is 100% one class
    2. When splitting a node will result in the tree exceeding a maximum depth (smaller tree, less prone to overfitting)
    3. Information gain from additional splits is less than threshold
    4. When number of examples in a node is below a threshold

Decision Tree Learning

- Start with all examples at the root node
- Calculate information gain for all possible features, and pick one with the highest info. gain
- Split dataset according to selected feature, and create left and right branches of tree
- Keep repeating splitting process until stopping criteria is met

One-hot encoding of categorical features: 1 when true

Tree ensembles 

Random forest

- A number of decision trees during training phase
- Use sampling with replacement to create a new training set of size m. Train a decision tree on the new dataset.
- Randomising the feature choice: At each node, when choosing a feature to use to split, if n features are available, pick a random subset of k < n features and allow the algorithm to only choose from that subset of features.

Boosted trees (XGBoost)

- Use sampling with replacement to create a new training set of size m. But instead of picking from all examples with equal (1/m) probability, make it more likely to pick misclassified examples from previously trained trees

Decision Tree & Tree ensembles

- Works well on tabular (structured) data
- Not recommended for unstructured data (images, audio, text)
- Fast
- Small decision trees may be human interpretable

Neural Networks

- Works well on all types of data, tabular (structured) and unstructured data
- May be slower than a decision tree
- Works with transfer learning
- When building a system of multiple models working together, it might be easier to string together multiple NN

## Unsupervised ML, Recommenders, Reinforcement Learning

Unsupervised ML

Clustering

- K-means
    - elbow method to find optimal number of clusters

Anomaly detection

- Gaussian (Normal) distribution
    - sigma-squared increase, flatter curve
    - mu (variance, standard deviation) increase, curve shifts right

Anomaly detection: future anomalies may look nothing like any of the anomalous examples we’ve seen so far, deviate from norm

Supervised learning: future positive examples likely to be similar to ones in training set

Recommender Systems

Collaborative filtering

- Based on ratings of users who gave similar ratings as you
- Cold start problem
    - how to rank new items that few users have rated
    - how to show something reasonable to new users who have rated few items

Content-based filtering

- Based on features of user and item to find good match
- Retrieval & ranking

Mean normalization

- feature scaling

Gradient descent algorithm implementation in TensorFlow

- Auto Diff / Auto Grad
    - tf.Variable : parameters we want to optimize
    - tf.GradientTape() : to record steps used to compute cost J to enable auto differentiation
    - tape.gradient() to calculate gradients of cost with respect to parameter
- Adam optimizer

PCA algorithm

- Preprocess features, normalized to have zero mean, feature scaling
- Reduce data to 1-dimension

Reinforcement Learning

- keywords: states, actions, rewards, discount factor, return, policy
- choose a policy pi(s) = a that will tell us what action a to take in state s so as to maximize the expect return

Markov Decision Process

- future depends only on current state, not how we reached our current state

State-action value function

- Q-function: Q(s, a), return if you start in state s, take action a (once) then behave optimally after that
- Best possible return from state s is max Q(s, a)
- Best possible action in state s is the action a that gives max Q(s, a)
- Bellman Equation: Q(s, a) = R(S) + y max Q(s’, a’), where y is discount factor, reward you get right away + return from behaving optimally starting from state s’

Epsilon-greedy policy

- Option 1:
    - pick action a that maximizes Q(s, a)
- Option 2:
    - with probability 0.95, pick action a that maximizes Q(s, a). Greedy, “Exploitation”
    - with probability 0.05, pick action a randomly. “Exploration”

Batch learning VS Mini-Batch learning

Soft Update