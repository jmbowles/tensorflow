import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

sess = tf.InteractiveSession()

#generating data
X_data=np.arange(0,100,0.1)
Y_data=X_data+20*np.sin(X_data/10)
#plotting the data
plt.scatter(X_data,Y_data)
#Uncomment below to see the plot of input data. 
#plt.show()
n_samples=1000
X_data=np.reshape(X_data,(n_samples,1))
Y_data=np.reshape(Y_data,(n_samples,1))
#batch size
batch_size=100

with tf.name_scope('input'):
    #placeholder for X_data
    X=tf.placeholder(tf.float32,shape=(batch_size,1), name='X')
    #placeholder for Y_data
    Y=tf.placeholder(tf.float32,shape=(batch_size,1), name='Y')
    #placeholder for checking the validity of our model after training
    X_check=tf.placeholder(tf.float32,shape=(n_samples,1), name='X_check')

#defining weight variable
with tf.name_scope('weights'):
    W=tf.Variable(tf.random_normal((1,1)), name='weights')
    variable_summaries(W)

#defining bias variable
with tf.name_scope('biases'):
    b=tf.Variable(tf.random_normal((1,)), name='bias')
    variable_summaries(b)

#generating predictions
with tf.name_scope('Wx_plus_b'):
    y_pred=tf.matmul(X,W)+b
    variable_summaries(y_pred)
    #tf.summary.histogram('y_pred', y_pred)
    

with tf.name_scope('loss'):
    #RMSE loss function
    loss=tf.reduce_sum(((Y-y_pred)**2)/batch_size)
    #defining optimizer
    opt_operation=tf.train.AdamOptimizer(.0001).minimize(loss)
    #tf.summary.histogram('loss', loss)
    variable_summaries(loss)

#creating a session object
#with tf.Session() as sess:
    #initializing the variables
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('logs/', sess.graph)

tf.global_variables_initializer().run()

#gradient descent loop for 500 steps
for iteration in range(20000):
    #selecting batches randomly
    indices=np.random.choice(n_samples,batch_size)
    X_batch,Y_batch=X_data[indices],Y_data[indices]
    #running gradient descent step
    summary, _, loss_value = sess.run([merged, opt_operation, loss],feed_dict={X:X_batch,Y:Y_batch})
    summary_writer.add_summary(summary, iteration)

summary_writer.close()
print(loss_value)

#plotting the predictions
y_check=tf.matmul(X_check,W)+b
pred=sess.run(y_check,feed_dict={X_check:X_data})
plt.scatter(X_data,pred)
plt.scatter(X_data,Y_data)
#plt.show()