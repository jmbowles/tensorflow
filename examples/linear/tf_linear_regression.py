import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_summary(summary_op):
  # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(summary_op)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(summary_op - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(summary_op))
    tf.summary.scalar('min', tf.reduce_min(summary_op))
    tf.summary.histogram('histogram', summary_op)

# The logging directory for tensorboard 
log_dir = 'logs/'

# Clean-up old tensorboard output
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

sess = tf.InteractiveSession()

# Create the data for X and Y. All 1000 values will b in one dimension (1000,)
X_data = np.arange(0,100,0.1)
Y_data = X_data+20*np.sin(X_data/10)

# Reshape the data into two dimensions (1000, 1). 1000 rows, 1 column
n_samples = 1000
X_data = np.reshape(X_data,(n_samples,1))
Y_data = np.reshape(Y_data,(n_samples,1))

# Batch size for training purposes
batch_size = 100

# Define the placeholders for the inputs that will be fed into the graph at runtime
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=(batch_size,1), name='X')
    Y = tf.placeholder(tf.float32, shape=(batch_size,1), name='Y')
    X_check = tf.placeholder(tf.float32, shape=(n_samples,1), name='X_check')

# Define the weight variable that will be learned at runtime
with tf.name_scope('weights'):
    W = tf.Variable(tf.random_normal((1,1)), name='weights')
    add_summary(W)

# Define the bias that will be learned at runtime
with tf.name_scope('biases'):
    b = tf.Variable(tf.random_normal((1,)), name='bias')
    add_summary(b)

# The linear function Y as a function of X 
with tf.name_scope('Wx_plus_b'):
    y_pred = tf.matmul(X,W)+b
    add_summary(y_pred)
    
# The gradient descent loss function to minimize, with the goal of optimizing W and b to produce y_pred
with tf.name_scope('loss'):
    # The standard RMSE loss function
    loss = tf.reduce_sum(((Y-y_pred)**2)/batch_size)
    optimizer = tf.train.AdamOptimizer(.0001).minimize(loss)
    add_summary(loss)

# Merge all the summaries for tensorboard output
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

tf.global_variables_initializer().run()

# Set the iterations for convergence using gradient descent
for iteration in range(20000):
    indices = np.random.choice(n_samples,batch_size)
    X_batch,Y_batch = X_data[indices],Y_data[indices]
    summary, _, loss_value = sess.run([merged, optimizer, loss], feed_dict={X:X_batch,Y:Y_batch})
    summary_writer.add_summary(summary, iteration)

summary_writer.close()
print(loss_value)

#plotting the predictions
y_check = tf.matmul(X_check,W)+b
pred = sess.run(y_check, feed_dict={X_check:X_data})
plt.scatter(X_data,pred)
plt.scatter(X_data,Y_data)
#plt.show()