#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:17:00 2018

@author: joaopedroaugusto
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
sns.set_style('dark')

import warnings
warnings.filterwarnings("ignore")

def f(x,y):
    return np.power(x,2)/2.0 + (2*np.power(y,2))

f = np.vectorize(f)

def plot_contour(f, from_, to_):
    fig = plt.figure(figsize=(17,7))
    s = np.linspace(from_, to_)
    X, Y = np.meshgrid(s,s)
    Z = f(X, Y)
    plt.contour(X, Y, Z, colors='b')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.hold(True)



def tfGradientDescent(start, alpha, nIter = 20, session=None):
    
    lr = tf.constant(alpha, dtype=tf.float32, name='learning_rate')
    x = tf.Variable(start, dtype=tf.float32, name='x')
    y = tf.pow(x[0],2)/2.0 + 2*tf.pow(x[1],2)
    
    grad = tf.gradients(y, x)[0]
    update = tf.assign(x, x - lr * grad)
    
    initializer = tf.global_variables_initializer()
    
    session = tf.Session() if session is None else session
    print(session)
    session.run(initializer)
    
    xs = [session.run(x)]
    grads = [session.run(grad)]
    ys = [session.run(y)]
    for _ in range(nIter):
        session.run(update)
        xs.append(session.run(x))
        grads.append(session.run(grad))
        ys.append(session.run(y))
    
    session.close()
    
    return xs, ys, grads

def tfMomentumGradientDescent(start, alpha, u, nIter = 20, session=None):
    
    lr = tf.constant(alpha, dtype=tf.float32, name='learning_rate')
    u = tf.constant(u, dtype=tf.float32, name='momentum')
    x = tf.Variable(start, dtype=tf.float32, name='x')
    v = tf.Variable(np.zeros(x.shape), dtype=tf.float32, name='v')
    y = tf.pow(x[0],2)/2.0 + 2*tf.pow(x[1],2)
    
    grad = tf.gradients(y, x)[0]
    updateV = tf.assign(v, u*v - lr * grad)
    update = tf.assign(x, x + v)
    
    initializer = tf.global_variables_initializer()
    
    session = tf.Session() if session is None else session
    session.run(initializer)
    
    xs = [session.run(x)]
    grads = [session.run(grad)]
    ys = [session.run(y)]
    for _ in range(nIter):
        session.run(updateV)
        session.run(update)
        xs.append(session.run(x))
        grads.append(session.run(grad))
        ys.append(session.run(y))
    
    session.close()
    
    return xs, ys, grads


def plotGD(xs,ys, grads, lr):
    
    plot_contour(f, xs[0][0]*1.1, xs[0][0]*-1.1)
    
    for i in range(len(xs)):
        x = xs[i]
        g = grads[i]
        
        # Adding dot
        plt.plot(x[0], x[1], 'ro')
        
        # Direction
        plt.arrow(x[0], x[1], -g[0], -g[1], linestyle='--', color='r', linewidth=0.1)
    
    xy=(xs[-1][0], xs[-1][1])
    plt.annotate("Last value: {0} \n f(x)={1:.2f}".format(np.round(xs[-1],2), ys[-1]), \
                 xy=xy, xytext=(50,-50), fontsize=14)
    plt.show()
    
alpha = 0.1
start = [-1000.0, -1000.0]
xs, ys, grads = tfGradientDescent(start, alpha, nIter=30)

plotGD(xs,ys, grads, alpha)

u = 0.5
xs, ys, grads = tfMomentumGradientDescent(start, alpha, u, nIter=30)
plotGD(xs,ys, grads, alpha)