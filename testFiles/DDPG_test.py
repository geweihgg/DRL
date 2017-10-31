#This file is a test file for DDPG.

from DDPG import ReplayBuffer

memory=ReplayBuffer(5)

memory.append([1,2],0,1,[1,3])
memory.append([2,3],1,1,[2,4])
memory.append([5,2],2,10,[6,2])
memory.append([2,2],1,2,[2,2])
s,a,r,s_=memory.sample_batch(3)
print s,a,r,s_

#for i in range(10):

#memory.append([1,2],0,1,[1,3])