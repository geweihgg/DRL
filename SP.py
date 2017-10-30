#!/usr/bin/python

#stochastic policy
#demostration for env pendulum

import gym

def main():
	env=gym.make('Pendulum-v0')
	
	print env.action_space
	print env.observation_space
	print env.action_space.high
	print env.action_space.low

	for i in range(10):
		env.reset()
		for j in range(200):
			env.render()
			# action=env.action_space.sample()
			action=[0.5]
			s,r,done,info=env.step(action)
			#print r
			print done
			if done:
				print ("Episode finished after {} timesteps".format(j+1))
				break



if __name__ == "__main__":
	main()