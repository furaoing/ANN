from Class_Definition_ANN import ANN
from dataset import data
from dataset import target
import pylab
import random

print("Please initialize the state of your neurons, the synaptic weights and the network biases before continue")

print("Program now initialize the network with default setting")

"""initialize network settings"""
network_depth=3			#set the depth of this network

network_width=1

alpha=1

target_var=1

n_input = 1
x=ANN(network_depth,network_width,target_var, alpha, n_input)

for j in range(network_width):
    x.Neu_hidden[0][j].initialize_synaptic_weight([1 for k in range(n_input)])
    x.Neu_hidden[0][j].initialize_bias(1)

for i in range(1, network_depth):
    for j in range(network_width):
        x.Neu_hidden[i][j].initialize_synaptic_weight([1 for k in range(network_width)])
        x.Neu_hidden[i][j].initialize_bias(1)

for i in range(target_var):
	x.Neu_output[i].initialize_synaptic_weight([1 for k in range(network_width)])
	x.Neu_output[i].initialize_bias(1)
"""initialize network setting"""

dev=list()
error=list()
for n in range(len(data)):
	error.append(100)
	
converged=False

#while not converged:
for i in range(1000):
	for n in range(len(data)):
		"""prepare training examples"""
		training_example=data[n]

		target_output=[target[n]]

		input_signal=training_example
		"""push training examples"""

		x.receive(input_signal,target_output)	#load input stream and target output into the network

		x.feed_forward()

		x.backpropagation()

		x.update_network_weight()
		
		error[n]=x.output_error
	
	# print(x.Neu_hidden[0][0].weight[0])
	
	all_error=sum(error)
	print(all_error)
	dev.append(all_error)
	
	#if all_error<2:
		#converged=True
"""
network_output=list()
_data = [0.6,0.1,0.9, 0.94, 0.2, 0.35]
for n in range(len(_data)):
		x.receive(_data,_data)	#load input stream and target output into the network

		x.feed_forward()

		network_output.append(x.network_output[0])
print(network_output)
"""
pylab.plot(dev)

pylab.show()
#print("Training Completed !")
