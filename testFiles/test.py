#!/usr/bin/python

#Test for single variable output
# import tensorflow as tf

# sess=tf.InteractiveSession()

# x=tf.placeholder(dtype=tf.float32,shape=[2,2])

# w=tf.constant(value=[2,3],dtype=tf.float32,shape=[2,1])

# x_input=[[1,5],[3,4]]

# y=tf.matmul(x,w)

# grad_y=tf.gradients(y,w)

# sess.run(tf.global_variables_initializer())

# print len(grad_y)    #len is 1
# print grad_y[0].eval(feed_dict={x:x_input})


#----------------------------------------------------------------------

#Test for multiple variable output 
# import tensorflow as tf
# sess=tf.InteractiveSession()

# x=tf.placeholder(dtype=tf.float32,shape=[2,2])

# w=tf.constant(value=[2,3,5,6],dtype=tf.float32,shape=[2,2])

# x_input=[[1,5],[3,4]]

# y=tf.matmul(x,w)

# grad_y=tf.gradients(y,w)

# sess.run(tf.global_variables_initializer())

# print len(grad_y)

# print grad_y[0].eval(feed_dict={x:x_input})

#-------------------------------------------------------------
#no batch
# import tensorflow as tf
# sess=tf.InteractiveSession()

# x=tf.placeholder(dtype=tf.float32,shape=[1,2])

# w=tf.constant(value=[2,3,5,6],dtype=tf.float32,shape=[2,2])

# x_input=[[1,5]]

# y=tf.matmul(x,w)

# grad_y=tf.gradients(y,w)

# sess.run(tf.global_variables_initializer())

# print len(grad_y)

# print grad_y[0].eval(feed_dict={x:x_input})

#-------------------------------------------------------------
#no batch, single variable output
# import tensorflow as tf
# sess=tf.InteractiveSession()

# x=tf.placeholder(dtype=tf.float32,shape=[1,2])

# w=tf.constant(value=[2,3],dtype=tf.float32,shape=[2,1])

# x_input=[[1,5]]

# y=tf.matmul(x,w)

# grad_y=tf.gradients(y,w)

# sess.run(tf.global_variables_initializer())

# print len(grad_y)

# print grad_y[0].eval(feed_dict={x: x_input})

#-------------------------------------------------------------
# import tensorflow as tf

# import numpy as np

# sess=tf.InteractiveSession()

# x=tf.placeholder(dtype=tf.float32,shape=[None,3])
# net=tf.layers.dense(x,5)
# y=tf.layers.dense(net,1)

# sess.run(tf.global_variables_initializer())

# inputs=[[1,2,3],[4,5,6]]

# print sess.run(y,feed_dict={x: inputs})

#-------------------------------------------------------------
# import tensorflow as tf

# with tf.Session() as sess:
	# x=tf.placeholder(dtype=tf.float32,shape=[None,3])
	# net=tf.layers.dense(x,5)
	# y=tf.layers.dense(net,1)
# 	writer=tf.summary.FileWriter('/home/moran/DRL/visualization',sess.graph)

#-------------------------------------------------------------
# import tensorflow as tf

# sess=tf.Session()

# x=tf.placeholder(dtype=tf.float32,shape=[None,3])
# w=tf.constant([1,2,3,4,5,6],dtype=tf.float32,shape=[2,3])
# y=x*w

# tf.summary.scalar("y",y)
# merged_summary=tf.summary.merge_all()

# sess.run(tf.global_variables_initializer())

# for i in range(100):
# 	inputs=[[i,i+1,i+2]]
# 	_,summary=sess.run([y,merged_summary],feed_dict={x: inputs})
# 	writer.add_summary(summary,i)
#-------------------------------------------------------------
import gym
# env = gym.make('CarRacing-v0')
# env=gym.make('LunarLanderContinuous-v2')
env=gym.make('BipedalWalker-v2')
# print env.observation_space
# print env.action_space

for i_episode in range(100):
	observation = env.reset()
	for t in range(1000):
		env.render()
		# print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

