import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt(r"C:\Users\jatin\Desktop\Study\Data\Mnist CSV\train.csv", delimiter=',', dtype=np.int32, skiprows=1)
m, n = data.shape  # m=42000, n=785
data.shape
np.random.shuffle(data)

data_vel = data[:1000].T
X_vel = data_vel[1:]
y_vel = data_vel[0]
X_vel = X_vel / 255.

data_train = data[1000:].T
X_train = data_train[1:]
X_train = X_train/255.
y_train = data_train[0]


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2


def ReLu(Z):
    return np.maximum(Z, 0)


def Softmax(Z):
    return np.exp(Z)/ sum(np.exp(Z))


def ReLU_deriv(Z):
    return Z > 0


def forward(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.imshow(current_image, interpolation='nearest', cmap="binary")
    plt.show()


W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 500)
test_prediction(50, W1, b1, W2, b2)
test_prediction(155, W1, b1, W2, b2)
test_prediction(244, W1, b1, W2, b2)
test_prediction(377, W1, b1, W2, b2)
