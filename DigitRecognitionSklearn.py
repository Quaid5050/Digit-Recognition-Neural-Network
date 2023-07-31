from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt

digits_data = load_digits()
digits = digits_data.data
targets = digits_data.target

x_train, x_test, y_train, y_test = train_test_split(digits, targets, test_size=0.25)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h))

from sklearn.preprocessing import OneHotEncoder

encode = OneHotEncoder(sparse=False)
Y = encode.fit_transform(y_train.reshape(-1, 1))

X = x_train
m = len(Y)
epochs = 100
B = np.zeros([10, 64])
alpha = 0.1

dumval = []
meanloss = []

for iteration in range(epochs):
    dB = np.zeros(B.shape)
    Loss = 0

    for j in range(X.shape[0]):
        x1 = X[j, :].reshape(64, 1)
        y1 = Y[j, :].reshape(10, 1)

        z1 = B.dot(x1)
        h = sigmoid(z1)

        db = (h - y1) * x1.T
        dB += db
        Loss += cost_function(h, y1)

    dB = dB / float(X.shape[0])
    Loss = Loss / float(X.shape[0])
    gradient = alpha * dB
    B = B - gradient
    dumval.append(Loss)
    meanloss.append(np.mean(Loss))

    print("Iteration:", iteration, "Loss:", Loss)

gray = cv2.imread("digitss/5.png", 0)
gray = cv2.resize(255 - gray, (8, 8))

g1 = gray.reshape(64, 1)
z1 = B.dot(g1)
h = sigmoid(z1)

predicted_digit = h.argmax(axis=0)[0]
print("Predicted Digit:", predicted_digit)
print("Probabilities:", h)

# Display the custom image and the predicted digit
plt.imshow(gray, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
