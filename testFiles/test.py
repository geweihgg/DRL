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
# import gym
# # env = gym.make('CarRacing-v0')
# # env=gym.make('LunarLanderContinuous-v2')
# env=gym.make('CarRacing-v0')
# print env.observation_space
# print len(env.observation_space.shape)
# print env.action_space.high
# print env.action_space.low
# bound=[list(env.action_space.high),list(env.action_space.low)]

# h_l=env.action_space.high-env.action_space.low
# print h_l


# print bound
# # observation=env.reset()
# # print observation[0]

# print env.action_space

# # for i_episode in range(100):
# # 	observation = env.reset()
# # 	for t in range(1000):
# # 		env.render()
# # 		# print(observation)
# # 		action = env.action_space.sample()
# # 		observation, reward, done, info = env.step(action)
# # 		if done:
# # 			print("Episode finished after {} timesteps".format(t+1))
# # 			break

#-------------------------------------------------------------
#test for gym wrappers










#-------------------------------------------------------------
# import tensorflow as tf

# sess=tf.InteractiveSession()

# a=tf.constant(value=[1,2,3,4,5,6],dtype=tf.float32,shape=[1,6])

# sess.run(tf.global_variables_initializer())

# print sess.run(a)

# b=tf.reshape(a,shape=[-1,3])

# print tf.shape(b)[0]

# print sess.run(b)

#-------------------------------------------------------------
import tensorflow as tf

sess=tf.InteractiveSession()

a=tf.constant(value=[1,2,3,4,5,6],dtype=tf.float32,shape=[2,3])

b=tf.constant(value=[4,5,6],dtype=tf.float32,shape=[1,3])
b=tf.tile(b,[tf.shape(a)[0],1])

c=tf.constant(value=[3,5,7],dtype=tf.float32,shape=[1,3])
print sess.run(b)

x=tf.constant(value=[1],dtype=tf.float32,shape=[1,1])
y=tf.constant(value=[1],dtype=tf.float32,shape=[1,1])
z=tf.constant(value=[1],dtype=tf.float32,shape=[1,1])
print sess.run(tf.concat([x[0],y[0],z[0]],0))

print sess.run(tf.multiply(a,b)+c)