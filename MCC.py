#!/usr/bin/python

import gym
import tensorflow as tf

env=gym.make('MountainCarContinuous-v0')

print env.observation_space
print env.action_space

with tf.Session() as sess:
	a=tf.constant(value=[1,2,3,4],dtype=tf.float32,shape=[2,2])
	print sess.run(a)
	