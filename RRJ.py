#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import time

# In[ ]:


def svd_product(A, U, S, VH): # A*U*S*VH
    Q, R = np.linalg.qr(np.matmul(A, U))
    u, s, vh = np.linalg.svd(np.matmul(R, np.diag(S)), full_matrices=False)
    return [np.matmul(Q,u), s, np.matmul(vh,VH)]

def svd_drei(A, B, C, D): # A*B*C*U*S*VH
    U, S, VH = np.linalg.svd(np.matmul(C, D), full_matrices=False)
    return svd_product(np.matmul(A, B), U, S, VH)

def Model(inputs, dimensionality, code_size1, code_size2):
    with tf.variable_scope('Autoencoder', reuse=tf.AUTO_REUSE) as scope:
        W1 = tf.get_variable('W1', initializer=tf.truncated_normal([dimensionality, code_size1], stddev=0.1))
        b1 = tf.get_variable('b1', initializer=tf.constant(0.1, shape=[code_size1]))
        W2 = tf.get_variable('W2', initializer=tf.truncated_normal([code_size1, code_size2], stddev=0.1))
        b2 = tf.get_variable('b2', initializer=tf.constant(0.1, shape=[code_size2]))
        W3 = tf.get_variable('W3', initializer=tf.truncated_normal([code_size2, code_size1], stddev=0.1))
        b3 = tf.get_variable('b3', initializer=tf.constant(0.1, shape=[code_size1]))
        W4 = tf.get_variable('W4', initializer=tf.truncated_normal([code_size1, dimensionality], stddev=0.1))
        b4 = tf.get_variable('b4', initializer=tf.constant(0.1, shape=[dimensionality]))
    code_data1 = tf.nn.sigmoid(tf.matmul(inputs, W1) + b1)
    code_data2 = tf.nn.sigmoid(tf.matmul(code_data1, W2) + b2)
    code_data3 = tf.nn.sigmoid(tf.matmul(code_data2, W3) + b3)
    code_data4 = tf.matmul(code_data3, W4) + b4
    return code_data4

def Gradients(inputs, dimensionality, code_size1, code_size2, batch_size):
    with tf.variable_scope('Autoencoder', reuse=tf.AUTO_REUSE) as scope:
        W1 = tf.get_variable('W1', initializer=tf.truncated_normal([dimensionality, code_size1], stddev=0.1))
        b1 = tf.get_variable('b1', initializer=tf.constant(0.1, shape=[code_size1]))
        W2 = tf.get_variable('W2', initializer=tf.truncated_normal([code_size1, code_size2], stddev=0.1))
        b2 = tf.get_variable('b2', initializer=tf.constant(0.1, shape=[code_size2]))
        W3 = tf.get_variable('W3', initializer=tf.truncated_normal([code_size2, code_size1], stddev=0.1))
        b3 = tf.get_variable('b3', initializer=tf.constant(0.1, shape=[code_size1]))
        W4 = tf.get_variable('W4', initializer=tf.truncated_normal([code_size1, dimensionality], stddev=0.1))
        b4 = tf.get_variable('b4', initializer=tf.constant(0.1, shape=[dimensionality]))
    code_data1 = tf.nn.sigmoid(tf.matmul(inputs, W1) + b1)
    code_data2 = tf.nn.sigmoid(tf.matmul(code_data1, W2) + b2)
    code_data3 = tf.nn.sigmoid(tf.matmul(code_data2, W3) + b3)

    grads = []
    for i in range(batch_size):
        code_data1_i = code_data1[i]
        sigma_prime1 = tf.multiply(1.0 - code_data1_i, code_data1_i)
        diag_sigma_prime1 = tf.linalg.diag(sigma_prime1)
        grad_1 = tf.matmul(W1, diag_sigma_prime1)

        code_data2_i = code_data2[i]
        sigma_prime2 = tf.multiply(1.0 - code_data2_i, code_data2_i)
        diag_sigma_prime2 = tf.linalg.diag(sigma_prime2)
        grad_2 = tf.matmul(W2, diag_sigma_prime2)

        code_data3_i = code_data3[i]
        sigma_prime3 = tf.multiply(1.0 - code_data3_i, code_data3_i)
        diag_sigma_prime3 = tf.linalg.diag(sigma_prime3)
        grad_3 = tf.matmul(W3, diag_sigma_prime3)

        grad_4 = W4
        
        grads.append(tf.matmul(grad_1, tf.matmul(grad_2, tf.matmul(grad_3, grad_4))))
    grads = tf.reshape(tf.stack(grads), [batch_size, dimensionality, dimensionality])
    return grads

def Drei(inputs, dimensionality, code_size1, code_size2, batch_size):
    with tf.variable_scope('Autoencoder', reuse=tf.AUTO_REUSE) as scope:
        W1 = tf.get_variable('W1', initializer=tf.truncated_normal([dimensionality, code_size1], stddev=0.1))
        b1 = tf.get_variable('b1', initializer=tf.constant(0.1, shape=[code_size1]))
        W2 = tf.get_variable('W2', initializer=tf.truncated_normal([code_size1, code_size2], stddev=0.1))
        b2 = tf.get_variable('b2', initializer=tf.constant(0.1, shape=[code_size2]))
        W3 = tf.get_variable('W3', initializer=tf.truncated_normal([code_size2, code_size1], stddev=0.1))
        b3 = tf.get_variable('b3', initializer=tf.constant(0.1, shape=[code_size1]))
        W4 = tf.get_variable('W4', initializer=tf.truncated_normal([code_size1, dimensionality], stddev=0.1))
        b4 = tf.get_variable('b4', initializer=tf.constant(0.1, shape=[dimensionality]))
    code_data1 = tf.nn.sigmoid(tf.matmul(inputs, W1) + b1)
    code_data2 = tf.nn.sigmoid(tf.matmul(code_data1, W2) + b2)
    code_data3 = tf.nn.sigmoid(tf.matmul(code_data2, W3) + b3)

    grads = []
    for i in range(batch_size):
        code_data1_i = code_data1[i]
        sigma_prime1 = tf.multiply(1.0 - code_data1_i, code_data1_i)
        grad_1 = sigma_prime1

        code_data2_i = code_data2[i]
        sigma_prime2 = tf.multiply(1.0 - code_data2_i, code_data2_i)
        grad_2 = sigma_prime2

        code_data3_i = code_data3[i]
        sigma_prime3 = tf.multiply(1.0 - code_data3_i, code_data3_i)
        grad_3 = sigma_prime3

        grads.append(grad_1)
        grads.append(grad_2)
        grads.append(grad_3)
    return grads


# In[ ]:


# iter_num number of iterations of alternating scheme
# steps_number number of gradient steps per iteration
# autoencoder has a neural network with architecture 
# [dimensionality, code_size1, code_size2, code_size1, dimensionality]
# k needed dimension
# gamma smoothness of manifold
# Lambda rank reducing parameter

def RRJ2(x_train, grad_x_train, k = 30, code_size1 = 120, code_size2 = 60, batch_size=20,          gamma = 1.0, epsilon = 0.1, Lambda=10.0, iter_num = 40, steps_number = 1000, learning_rate = 0.0001, new_curvature = False):
    N = x_train.shape[0]
    dimensionality = x_train.shape[1]
    N_grad = grad_x_train.shape[0]
    if ((N%batch_size != 0) or (N_grad%batch_size != 0)):
        print("It is much better if batch is a divisor of both x_train.shape[0] and grad_x_train.shape[0]\n")

        
    # Define placeholders
    tf.reset_default_graph() 
    training_data = tf.placeholder(tf.float32, [None, dimensionality])
    gradient_training_data = tf.placeholder(tf.float32, [None, dimensionality])
    old_P = tf.placeholder(tf.float32, shape=[None, dimensionality, dimensionality])
    
    recover = Model(training_data, dimensionality, code_size1, code_size2)
    grad_phi_psi = Gradients(training_data, dimensionality, code_size1, code_size2, batch_size)
    # this is gradient field close to our points
    noise = tf.random.normal(shape=[batch_size, dimensionality],mean=0.0,stddev=epsilon)
    rand_training_data = training_data + noise
    new_grad_phi_psi = Gradients(gradient_training_data, dimensionality, code_size1, code_size2, batch_size)

    grad_drei = Drei(gradient_training_data, dimensionality, code_size1, code_size2, batch_size)

    # Define the loss function
    if new_curvature:
        rand_recover = Model(rand_training_data, dimensionality, code_size1, code_size2)
        curvatures = []
        for i in range(batch_size):
            grad_epsilon = tf.matmul(tf.reshape(noise[i], [1, -1]), grad_phi_psi[i])
            curv = tf.sqrt(tf.reduce_sum(tf.square(rand_recover[i]-recover[i]-\
                    grad_epsilon)))/(tf.reduce_sum(tf.square(grad_epsilon))+0.0001)
            curvatures.append(curv)
        curvature = tf.reduce_mean(tf.reshape(tf.stack(curvatures), [batch_size]))
        loss = tf.reduce_mean(tf.square(training_data - recover)) + gamma*curvature+\
               Lambda*tf.reduce_mean(tf.square(new_grad_phi_psi - old_P))
    else:
        rand_grad_phi_psi = Gradients(rand_training_data, dimensionality, code_size1, code_size2, batch_size)
        loss = tf.reduce_mean(tf.square(training_data - recover)) +            gamma*tf.reduce_mean(tf.square(grad_phi_psi-rand_grad_phi_psi)) +            Lambda*tf.reduce_mean(tf.square(new_grad_phi_psi - old_P))
    
    # Training step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # Run the training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())    
    
    cur_U = np.zeros((N_grad, dimensionality, k))
    cur_Sigma = np.zeros((N_grad, k, k))
    cur_V = np.zeros((N_grad, k, dimensionality))
    feed_P = np.zeros((batch_size, dimensionality, dimensionality))
    feed_P_all = np.zeros((N_grad, dimensionality, dimensionality))
    cur_W1 = np.random.normal(0, 0.35, (dimensionality, code_size1))
    cur_b1 = np.zeros((code_size1))
    cur_W2 = np.random.normal(0, 0.35, (code_size1, code_size2))
    cur_b2 = np.zeros((code_size2))
    cur_W3 = np.random.normal(0, 0.35, (code_size2, code_size1))
    cur_b3 = np.zeros((code_size1))
    cur_W4 = np.random.normal(0, 0.35, (code_size1, dimensionality))
    cur_b4 = np.zeros((dimensionality))
    
    num_batches = int(N/batch_size)
    grad_num_batches = int(N_grad/batch_size)
    
    I_cur = 1
    
    for iter in range(iter_num):
        start = time. time()
        I_cur = I_cur-1
        for i in range(steps_number):
            # Get the next batch
            which_batch = I_cur%num_batches
            input_batch = x_train[which_batch*batch_size:(which_batch+1)*batch_size]
            grad_which_batch = I_cur%grad_num_batches
            grad_input_batch = grad_x_train[grad_which_batch*batch_size:(grad_which_batch+1)*batch_size]
            for b in range(batch_size):
                U = cur_U[grad_which_batch*batch_size+b]
                Sigma = cur_Sigma[grad_which_batch*batch_size+b]
                V = cur_V[grad_which_batch*batch_size+b]
                feed_P[b] = np.matmul(U,np.matmul(Sigma,V))
            feed_dict = {training_data: input_batch, gradient_training_data: grad_input_batch, 
                         old_P:feed_P}
            # Run the training step
            train_step.run(feed_dict=feed_dict)
            # Print the accuracy progress on the batch every 100 steps
            if (i%200 == 0) or (i==steps_number-1):
                train_accuracy = sess.run(loss, feed_dict=feed_dict)
                print("Step %d, training batch accuracy %g %%"%(i, train_accuracy*100))
            I_cur = I_cur+1
        with tf.variable_scope('Autoencoder', reuse=tf.AUTO_REUSE) as scope:
            W10 = tf.get_variable('W1')
            b10 = tf.get_variable('b1')
            W20 = tf.get_variable('W2')
            b20 = tf.get_variable('b2')
            W30 = tf.get_variable('W3')
            b30 = tf.get_variable('b3')
            W40 = tf.get_variable('W4')
            b40 = tf.get_variable('b4')
        [cur_W1, cur_b1, cur_W2, cur_b2, cur_W3, cur_b3, cur_W4, cur_b4] =                       sess.run([W10, b10, W20, b20, W30, b30, W40, b40])
        stop = time. time()
        print("The time of the gradient descent part:", stop - start)
        start = time. time()
        for grad_which_batch in range(grad_num_batches):
            grad_input_batch = grad_x_train[grad_which_batch*batch_size:(grad_which_batch+1)*batch_size]
            feed_dict = {gradient_training_data: grad_input_batch}
            local_grad = sess.run(grad_drei, feed_dict=feed_dict)
            for b in range(batch_size):
                A = local_grad.pop(0)
                B = local_grad.pop(0)
                C = local_grad.pop(0)
                u, s, vh = svd_drei(np.matmul(cur_W1, np.diag(A)), np.matmul(cur_W2, np.diag(B)), np.matmul(cur_W3, np.diag(C)), cur_W4)
                cur_U[grad_which_batch*batch_size+b] = u[:,0:k:1]
                cur_V[grad_which_batch*batch_size+b] = vh[0:k:1,:]
                cur_Sigma[grad_which_batch*batch_size+b] = np.diag(s[0:k:1])
        stop = time. time()
        print("The time of the svd part:", stop - start)
    Autoencoder = [cur_W1, cur_b1, cur_W2, cur_b2, cur_W3, cur_b3, cur_W4, cur_b4]
    Tangent = np.zeros((N, k, dimensionality))
    for which_batch in range(num_batches):
        input_batch = x_train[which_batch*batch_size:(which_batch+1)*batch_size]
        feed_dict = {gradient_training_data: input_batch}
        local_grad = sess.run(grad_drei, feed_dict=feed_dict)
        for b in range(batch_size):
            A = local_grad.pop(0)
            B = local_grad.pop(0)
            C = local_grad.pop(0)
            u, s, vh = svd_drei(np.matmul(cur_W1, np.diag(A)), np.matmul(cur_W2, np.diag(B)), np.matmul(cur_W3, np.diag(C)), cur_W4)
            Tangent[which_batch*batch_size+b] = vh[0:k:1,:]
    sess.close()
    return Autoencoder, Tangent

