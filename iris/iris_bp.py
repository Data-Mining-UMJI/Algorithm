import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# train data

localfn='train.csv'
# read the first 4 columns
x_data = np.genfromtxt(localfn,delimiter=',',usecols=(0,1,2,3)) 
print(x_data.shape)
# read the fifth column
target = np.genfromtxt(localfn,delimiter=',',usecols=(4))


y_data = np.zeros((len(target),3))
#type(t) #show type of t (numpy.ndarray)
#print t #show contains of t

y_data[target == 0] = [1,0,0]
y_data[target == 1] = [0,1,0]
y_data[target == 2] = [0,0,1]
print(y_data.shape)
# set placeholder to receive data
# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 3])

# define the layers
# add hidden layer inpuy : xs 12unit   
l1 = add_layer(xs, 4, 12, activation_function=tf.nn.relu)
# add output layer input: l1  output  3 classes
prediction1 = add_layer(l1, 12, 3, activation_function=None)
# add softmax layer
prediction = tf.nn.softmax(prediction1)


# define loss
# the error between prediciton and real data    
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                              reduction_indices=[1]))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))


# choose optimizer     
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


# important step, initial
init = tf.initialize_all_variables()
sess = tf.Session()
# run
sess.run(init)

# sess.run optimizer
for i in range(5000):
    # training train_step & loss , placeholder ,feed give data
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 20 == 0:
        # to see the step improvement
        print(sess.run(cross_entropy, feed_dict={xs: x_data, ys: y_data}))
#=======================================================================================
# Test model
test='test.csv'
# read the first 4 columns
x_test = np.genfromtxt(test,delimiter=',',usecols=(0,1,2,3)) 
# read the fifth column
target_test = np.genfromtxt(test,delimiter=',',usecols=(4))


y_test = np.zeros((len(target_test),3))
#type(t) #show type of t (numpy.ndarray)
#print t #show contains of t

y_test[target_test == 0] = [1,0,0]
y_test[target_test == 1] = [0,1,0]
y_test[target_test == 2] = [0,0,1]

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print("Result:", sess.run(tf.argmax(prediction,1), feed_dict={xs: x_test}))
print("Accuracy:", sess.run(accuracy, feed_dict={xs: x_test, ys: y_test}))
