import gzip
import pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h



f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print(train_y[57])
"""


# TODO: the neural net!!
y_data = one_hot(train_y.astype(int), 10)
y_dataVal = one_hot(valid_y.astype(int), 10)
y_dataTest = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

w1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

w2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


h = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
y = tf.nn.softmax(tf.matmul(h, w2) + b2)


loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 100
valError0=0
valError1=0
a_valError=[]
a_trainError=[]
for epoch in range(100):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    trainError = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    a_trainError.append(trainError)
    actualValError = sess.run(loss, feed_dict={x: valid_x, y_: y_dataVal})
    print("EL ERROR DE VALIDACION -->", actualValError)
    if epoch==0:
        valError0; valError1 = actualValError

    else:

        valError0=valError1
        valError1=actualValError
    a_valError.append(actualValError)
    print("Epoch #:", epoch, "Error: ", trainError)
    result = sess.run(y, feed_dict={x: batch_xs})

    """
        for b, r in zip(batch_ys, result):
            print(b, "-->", r)
        print("----------------------------------------------------------------------------------")
    """

    if abs(valError1-valError0)/valError0 < 0.00001:
        print("------- Resultados Finales -------")
        print("Epoch: #", epoch)
        print("Error de Validacion Final ---> ", valError1)
        print("Error de Entrenamiento Final --->", trainError)
        break





print("----------------------")
print("      Start Test...   ")
print("----------------------")

aciertos = 0
totales = len(test_x)
for index in range(int(len(test_x)/batch_size)):
    batch_testX = test_x[index*batch_size : index*batch_size + batch_size]
    batch_testY = y_dataTest[index*batch_size : index*batch_size + batch_size]
    sess.run(train, feed_dict={x: batch_testX, y_ : batch_testY})
    result = sess.run(y, feed_dict={x: batch_testX})
    for index2, (a, b) in enumerate(zip(result, batch_testY)):
        if(np.argmax(result[index2]) == np.argmax(batch_testY[index2])):
            aciertos+=1

print("Aciertos ---> ",aciertos)
print("Totales ---> ",totales)
print("Porcentaje Acierto/Total ---> ",(aciertos/totales)*100)

print("\n===================================\n")

print("----------------------")
print("   Start Ploting...  ")
print("----------------------")


import matplotlib.pyplot as plt
plt.ioff()
plt.subplot(1,2,1)
plt.plot(a_trainError)
plt.xlabel("Time (epochs)")
plt.ylabel("Error")
plt.title("Error de Entrenamiento")

plt.subplot(1,2,2)
plt.xlabel("Time (epochs)")
plt.ylabel("Error")
plt.title("Error de Validacion")
plt.plot(a_valError, 'g')

plt.savefig("figuras/Figura3.jpeg")
plt.show()

