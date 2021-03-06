#!/usr/bin/python
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

DDPG is Actor Critic Based Algorithm.

Use raw pixels as inputs.
Only use batch normalization for input layers.

This version only solve the CarRacing-v0 env.

<Continuous Control with Deep Reinforcement Learning>

using:
tensorflow 1.2.1
numpy 1.13.1
tflearn 0.3
gym 0.9.3
"""
import random
import argparse
from collections import deque
from collections import namedtuple 
import tensorflow as tf
import numpy as np
import tflearn
import gym
from gym import wrappers
import pprint as pp

###############################  Actor  ####################################

class Actor(object):
	"""
	input: state
	output: action
	under a deterministic policy.
	"""
	def __init__(self,sess,state_dim,action_dim,action_bound,learning_rate,tau,BATCH_SIZE):
		"""
		The initializer for actor.
		state_dim: the input dim of the network
		action_dim: the output dim of the network
		action_bound: the bound used for clipping the output of the network
		tau: the ratio for update the target networks
		BATCH_SIZE: used for computing the average of gradients over the batch
		"""
		self.sess=sess
		self.state_dim=state_dim
		self.action_dim=action_dim
		self.action_bound=action_bound
		self.learning_rate=learning_rate
		self.tau=tau

		#behavior network
		self.inputs,self.outputs,self.scaled_out=self.create_actor_network()
		self.network_params=tf.trainable_variables()

		#target network
		self.target_inputs,self.target_outputs,self.target_scaled_out=self.create_actor_network()
		self.target_network_params=tf.trainable_variables()[len(self.network_params):]

		#Op for updating target network
		self.soft_update=[tf.assign(t,(self.tau*b+(1-self.tau)*t)) \
		        for b,t in zip(self.network_params,self.target_network_params)]

		#This gradient will be provided by the  critic network(behavior network).
		self.action_gradient=tf.placeholder(tf.float32,[None,self.action_dim])    #[None, self.action_dim]
		#Use the gradient provided by the cirtic network to weight the gradients of actor network(behavior network).
		#action_gradient: [None, self.action_dim]
		#scaled_out: [None, self.action_dim]
		#network_params: All trainable variables in behavior actor network, for example, 3.
		#actor_gradients: [[None,self.action_dim], [None,self.action_dim], [None,self.action_dim]]
		#TODO: We should divide self.actor_gradients by N, or divide the learning rate by N.
		self.actor_gradients=tf.gradients( \
			self.scaled_out,self.network_params,self.action_gradient)

		#Optimization Op
		#-self.learning_rate for ascent policy
		opt=tf.train.AdamOptimizer(-self.learning_rate/BATCH_SIZE)
		self.optimize=opt.apply_gradients(zip(self.actor_gradients,self.network_params))

		self.num_trainable_vars=len(self.network_params)+len(self.target_network_params)

	def create_actor_network(self):
		"""
		Create a conv network for raw-pixel inputs.
		Three conv layers wihout pooling + Two fully connected layers.
		"""

		#self.state_dim: (96,96,3)
		state_dim=self.state_dim
		inputs=tflearn.input_data(shape=[None,state_dim[0],state_dim[1],state_dim[2]])
		# net=tflearn.layers.normalization.batch_normalization(inputs)
		net=inputs

		#conv_2d(weights_init='uniform')-->Uniform[0,1]
		#It's too large, thus we may get 'nan' outputs
		#So I use the same init as the last layer mentioned in the paper
		#So as to ensure the init outputs of the actor/critic network near zero
		#But for the actor network, if it's too small, the agent will be overcautious,
		#and then it will take a long time to learn sth. 
		minv=-3e-4
		maxv=3e-4

		weight_init_conv1=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		weight_init_conv2=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		weight_init_conv3=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		weight_init_fc1=tflearn.initializations.uniform(minval=minv,maxval=maxv)

		net=tflearn.conv_2d(net,32,5,activation='relu',weights_init=weight_init_conv1,name='conv1')
		net=tflearn.conv_2d(net,32,5,activation='relu',weights_init=weight_init_conv2,name='conv2')
		net=tflearn.conv_2d(net,32,5,activation='relu',weights_init=weight_init_conv3,name='conv3')
		net=tflearn.fully_connected(net,200,weights_init=weight_init_fc1,name='fc1')

		w_init_s=tflearn.initializations.uniform(minval=-3e-4,maxval=3e-4)
		w_init_g=tflearn.initializations.uniform(minval=-3e-4,maxval=3e-4)
		w_init_b=tflearn.initializations.uniform(minval=-3e-4,maxval=3e-4)

		steer=tflearn.fully_connected(net,1,activation='tanh',weights_init=w_init_s)
		gas=tflearn.fully_connected(net,1,activation='sigmoid',weights_init=w_init_g)
		brake=tflearn.fully_connected(net,1,activation='sigmoid',weights_init=w_init_b)

		outputs=tf.concat([steer,gas,brake],1)
		scaled_out=outputs

		# #if we combine these action outputs togther, we'll have some problems.
		# outputs=tflearn.fully_connected(net,self.action_dim,activation='tanh',weights_init=w_init)
		# #action_space.high-action_space.low
		# h_l=tf.constant(value=(self.action_bound[0]-self.action_bound[1]),dtype=tf.float32,shape=[1,self.action_dim])
		# print self.sess.run(h_l)
		# #tile the h_l to [None,self.action_dim]
		# #output-->[None,self.action_dim]
		# h_l_tile=tf.tile(h_l,[tf.shape(outputs)[0],1])
		# #(h+l)/2
		# h_l_ave=tf.constant(value=((self.action_bound[0]+self.action_bound[1])/2.0),dtype=tf.float32,shape=[1,self.action_dim])
		# print self.sess.run(h_l_ave)
		# scaled_out=tf.multiply(tf.div(outputs,2.0),h_l_tile)+h_l_ave
		
		return inputs,outputs,scaled_out

	def train(self,inputs,a_gradient):
		self.sess.run(self.optimize,feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def predict(self,inputs):
		return self.sess.run(self.scaled_out,feed_dict={
			self.inputs: inputs
		})

	def predict_target(self,inputs):
		return self.sess.run(self.target_scaled_out,feed_dict={
			self.target_inputs: inputs
		})

	def update_target_network(self):
		self.sess.run(self.soft_update)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars


class Critic(object):
	"""
	input: state,action
	output: Q(s,a)
	The action must be obtained from the output of the actor network.
	"""
	def __init__(self,sess,state_dim,action_dim,learning_rate,tau,gamma,num_actor_vars,weight_decay):
		"""
		The initializer for critic.
		tau: the ratio for update the target networks
		gamma: the factor used in the computation of target y_i
		weight decay: the weight decay params
		"""
		self.sess=sess
		self.state_dim=state_dim
		self.action_dim=action_dim
		self.learning_rate=learning_rate
		self.tau=tau
		self.gamma=gamma
		self.weight_decay=weight_decay

		#behavior network
		self.inputs,self.actions,self.outputs=self.create_critic_network()
		self.network_params=tf.trainable_variables()[num_actor_vars:]

		#target network
		self.target_inputs,self.target_actions,self.target_outputs=self.create_critic_network()
		self.target_network_params=tf.trainable_variables()[(len(self.network_params)+num_actor_vars):]

		self.soft_update=[tf.assign(t,(self.tau*b+(1-self.tau)*t)) \
		        for b,t in zip(self.network_params,self.target_network_params)]

		#Network target(y_i)
		self.predicted_q_value=tf.placeholder(tf.float32,shape=[None,1])

		#Define loss and optimization Op
		#gradient descent
		#This can automatically update the changeable variables,
		#thus there is no need to use apply_gradients
		self.loss=tflearn.mean_square(self.predicted_q_value,self.outputs)
		self.optimize=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		#Get the gradient of the net w.r.t. the action.
		
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		# This means that when we use this function to compute gradients wrt. 
		# the input, it won't sum up over the batch.
		#This is not the same case as in the actor's gradients.
		self.action_gradient=tf.gradients(self.outputs,self.actions)

	def create_critic_network(self):
		state_dim=self.state_dim
		inputs=tflearn.input_data(shape=[None,state_dim[0],state_dim[1],state_dim[2]]) 
		actions=tflearn.input_data(shape=[None,self.action_dim])    #[None, self.action_dim]

		net=tflearn.layers.normalization.batch_normalization(inputs)

		#conv_2d(weights_init='uniform')-->Uniform[0,1]
		#It's too large, thus we may get 'nan' outputs
		#So I use the same init as the last layer mentioned in the paper
		#So as to ensure the init outputs of the actor/critic network near zero		
		minv=-3e-4
		maxv=3e-4
		weight_init_conv1=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		weight_init_conv2=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		weight_init_conv3=tflearn.initializations.uniform(minval=minv,maxval=maxv)

		net=tflearn.conv_2d(net,32,5,activation='relu',weights_init=weight_init_conv1,name='conv1',regularizer='L2',weight_decay=self.weight_decay)
		net=tflearn.conv_2d(net,32,5,activation='relu',weights_init=weight_init_conv2,name='conv2',regularizer='L2',weight_decay=self.weight_decay)
		net=tflearn.conv_2d(net,32,5,activation='relu',weights_init=weight_init_conv3,name='conv3',regularizer='L2',weight_decay=self.weight_decay)

		#Add the action tensor in the 2nd hidden layer
		#Use two temp layers to get the corresponding weights and biases
		#We can use tflearn.layers.merge_ops.merge to do this too.
		weight_init_fc11=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		weight_init_fc12=tflearn.initializations.uniform(minval=minv,maxval=maxv)
		t1=tflearn.fully_connected(net,200,weights_init=weight_init_fc11,regularizer='L2',weight_decay=self.weight_decay)
		t2=tflearn.fully_connected(actions,200,weights_init=weight_init_fc12,regularizer='L2',weight_decay=self.weight_decay)
		#net-->[None,96,96,32]
		#t1.W-->[294912,200]
		#96*96*32=294912
		#So we should reshape the tensor 'net' to [None,294912]
		#The order of the input sequence [None, 294912] dosen't matters.
		net_reshape=tf.reshape(net,shape=[-1,tf.shape(t1.W)[0]])    #[None,294912]
		net=tflearn.activations.relu(tf.matmul(net_reshape,t1.W)+tf.matmul(actions,t2.W))

		w_init=tflearn.initializations.uniform(minval=-3e-4,maxval=3e-4)
		outputs=tflearn.fully_connected(net,1,weights_init=w_init,regularizer='L2',weight_decay=self.weight_decay)
		return inputs,actions,outputs

	def train(self,inputs,actions,predicted_q_value):
		return self.sess.run([self.outputs,self.optimize],feed_dict={
			self.inputs: inputs,
			self.actions: actions,
			self.predicted_q_value: predicted_q_value    #predicted_q_value means y_i, it's the target q value
		})

	def predict(self,inputs,actions):
		return self.sess.run(self.outputs,feed_dict={
			self.inputs: inputs,
			self.actions: actions
		})

	def predict_target(self,inputs,actions):
		return self.sess.run(self.target_outputs,feed_dict={
			self.target_inputs: inputs,
			self.target_actions: actions
		})

	def update_target_network(self):
		self.sess.run(self.soft_update)

	def action_gradients(self,inputs,actions):
		return self.sess.run(self.action_gradient,feed_dict={
			self.inputs: inputs,
			self.actions: actions
		})


class ReplayBuffer(object):
	def __init__(self,capacity,random_seed=123):
		"""
		Initializer, 'capacity' means the maximum buffer size, 
		"""
		self.capacity=capacity
		self.count=0
		self.buffer=deque()
		self.dtype=namedtuple('Transition',['s','a','r','t','s_'])    #Data type for the replay buffer
		#Set the random seed, thus we can have different permutations of the transition.
		random.seed(random_seed)

	def size(self):
		return self.count

	def append(self,s,a,r,t,s_):
		experience=self.dtype(s,a,r,t,s_)
		if self.count<self.capacity:
			self.buffer.append(experience)
			self.count+=1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def sample_batch(self,batch_size):
		"""
		batch_size specifies the number of experience to add
		to the batch. If the replay buffer has less than batch_size
		elements, throw out exception.
		"""
		batch=[]

		assert self.count>=batch_size, 'The total size of buffer is not greater than batch size'

		batch=random.sample(self.buffer,batch_size)

		s=np.array([_.s for _ in batch])
		a=np.array([_.a for _ in batch])
		r=np.array([_.r for _ in batch])
		t=np.array([_.t for _ in batch])
		s_=np.array([_.s_ for _ in batch])

		return s,a,r,t,s_

	def clear(self):
		self.buffer.clear()
		self.count=0

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
	def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
		        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def build_summaries():
	episode_reward = tf.Variable(0.)
	tf.summary.scalar("Reward", episode_reward)
	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("Qmax Value", episode_ave_max_q)

	summary_vars = [episode_reward, episode_ave_max_q]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars


def train(sess,env,args,actor,critic,actor_noise):
	summary_ops,summary_vars=build_summaries()

	sess.run(tf.global_variables_initializer())

	#Save the sess.graph
	#summary_dir: default='./results/tf_ddpg'
	writer=tf.summary.FileWriter(args['summary_dir'],sess.graph)

	test_temp=[]
	for i in range(5):
		test_temp.append(1)



	#Initialize the target network.
	actor.update_target_network()
	critic.update_target_network()

	#Initialize the replay buffer.
	replay_buffer=ReplayBuffer(int(args['buffer_size']),int(args['random_seed']))

	for i in range(int(args['max_episodes'])):    #'for loop' for episode
		s=env.reset()    #The initial observation

		ep_reward=0        #What's this? This means that the total reward in this episode.
		ep_ave_max_q=0    #What's this?  Sum up max(Q(s,a)) in every step and average over total episode.

		for j in range(int(args['max_step_num'])):    #'for loop' for steps in one episode
			if args['render_env']:
				env.render()

			#Select an action using the behavior actor network, and add exploration noise to it.
			#actor_noise(type OrnsteinUhlenbeckActionNoise) is initialized in the main function.
			#batch_size=1, thus (1,actor.state_dim)
			#adim_t means actor state dim temp
			adim_t=actor.state_dim
			a=actor.predict(np.reshape(s,(1,adim_t[0],adim_t[1],adim_t[2])))+actor_noise()    #not a batch, just run an episode
			print a

			#For example, input=[[1,2,3]], output=[[0]], so we should use a[0]
			s_,r,terminal,info=env.step(a[0])
			
			#The shape of a is [batch_size,1,1], so we should reshape this
			#replay_buffer.append(s,a,r,terminal,s_)
			replay_buffer.append(np.reshape(s, (adim_t[0],adim_t[1],adim_t[2],)), np.reshape(a, (actor.action_dim,)),  \
				r, terminal, np.reshape(s_, (adim_t[0],adim_t[1],adim_t[2],)))


			if replay_buffer.size()>int(args['batch_size']):
				batch_s,batch_a,batch_r,batch_t,batch_s_=\
				    replay_buffer.sample_batch(int(args['batch_size']))

				#Calculate the target Q batch
				target_q=critic.predict_target(batch_s_,actor.predict_target(batch_s_))
				# print actor.predict_target(batch_s_)[0]

				y_i=[]

				for k in range(int(args['batch_size'])):
					if batch_t[k]:
						y_i.append(batch_r[k])
					else:
						y_i.append(batch_r[k]+critic.gamma*target_q[k])

				#Update the behavior critic network by minimizing the loss
				#Use the network before update to predict Q values, that is predicted_Q_value
				y_i_shaped=np.reshape(y_i,(int(args['batch_size']),1))
				# #Test code:
				# print y_i_shaped
				# print sess.run(tf.shape(y_i_shaped))
				# print y_i_shaped.size
				# print sess.run(tf.shape(batch_s))    #[64,3] 
				# print sess.run(tf.shape(batch_a))    #[64,1,1]
				# print test_temp

				predicted_Q_value,_=critic.train(batch_s,batch_a,y_i_shaped)

				#Use the predicted_q_value to compute the max Q, max over the batch
				#Average is done in summary step
				ep_ave_max_q+=np.amax(predicted_Q_value)

				#Update the actor policy using the sampled gradient
				a_outputs=actor.predict(batch_s)

				a_gradient=critic.action_gradients(batch_s,a_outputs)
				actor.train(batch_s,a_gradient[0])

				#Update target networks
				actor.update_target_network()
				critic.update_target_network()

			s=s_
			ep_reward+=r

			#Pendulum env won't be done until 200 steps, thus the 'ep_reward' is the total rewards of 200 steps.
			if terminal:
				summary_str=sess.run(summary_ops,feed_dict={
					summary_vars[0]:ep_reward,
					summary_vars[1]:ep_ave_max_q/float(j)
				})

				writer.add_summary(summary_str,i)
				writer.flush()

				print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
					i, (ep_ave_max_q / float(j))))

				break


def main(args):
	with tf.Session() as sess:
		env=gym.make(args['env'])
		np.random.seed(int(args['random_seed']))
		tf.set_random_seed(int(args['random_seed']))
		env.seed(int(args['random_seed']))

		#While in low-dim problem, state_dim=env.observation_space.shape[0]
		#Cause it's Box(3,)
		#But for raw-pixel inputs, it's Box(96,96,3)
		state_dim=env.observation_space.shape
		#action_dim--> the same as low-dim
		action_dim=env.action_space.shape[0]
		#TODO:
		action_bound=[env.action_space.high,env.action_space.low]

		#For low-dim problem
		#Ensure action bound is symmetric
		#Cause this is the minimal feasible version
		# assert(env.action_space.high==-env.action_space.low)
		#For raw-pixel inputs
		#Nothing to assert

		actor=Actor(sess,state_dim,action_dim,action_bound,float(args['actor_lr']),float(args['tau']),float(args['batch_size']))

		critic=Critic(sess,state_dim,action_dim,float(args['critic_lr']),float(args['tau']), \
			float(args['gamma']),actor.get_num_trainable_vars(),float(args['weight_decay']))

		#According to the paper, mu is zero
		actor_noise=OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

		#TODO: Wrappers
		if args['use_gym_monitor']:
			if not args['render_env']:    #render_env=False
				env=wrappers.Monitor( \
					env, args['monitor_dir'],video_callable=False,force=True)
		else:
			env=wrappers.Monitor(env,args['monitor_dir'],force=True)

		train(sess,env,args,actor,critic,actor_noise)

		#TODO: Wrappers close, this will lead to some errors
		#We can fix it by updating gym, but now I just comment it out.
		# if args['use_gym_monitor']:
		# 	env.monitor.close()

if __name__ == "__main__":

	#File dir
	path='CarRacing/exp1_ddpg_monitor'

	parser=argparse.ArgumentParser(description='provide arguments for DDPG agent')

	# agent parameters, default arguments are provided according to the paper
	parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
	parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
	parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
	parser.add_argument('--tau', help='soft target update parameter', default=0.001)
	parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
	parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=64)

	#L2 weight decay, weight decay params
	#TODO
	parser.add_argument('--weight-decay',help='weight decay of critic network', default=0.01)

	# run parameters
	parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
	parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
	parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
	parser.add_argument('--max-step-num', help='max step of 1 episode', default=1000)
	parser.add_argument('--render-env', help='render the gym env', action='store_true')
	parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
	parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./monitor/'+path)
	parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./tensorboard/'+path)

	#We use the gym wrappers to save the training records
	parser.set_defaults(render_env=True)
	parser.set_defaults(use_gym_monitor=True)

	args = vars(parser.parse_args())

	pp.pprint(args)

	main(args)

