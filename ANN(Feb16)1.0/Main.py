from Class_Definition_ANN import ANN

"""prepare training examples"""
training_example=[0,0,1]

target_output=[3]

input_signal=training_example
"""push training examples"""

print("Please initialize the state of your neurons, the synaptic weights and the network biases before continue")

print("Program now initialize the network with default setting")

"""initialize network settings"""
network_depth=3			#set the depth of this network

network_width=3

target_var=len(target_output)

x=ANN(network_depth,network_width,target_var)

for i in range(network_depth):
	for j in range(network_width):
		x.Neu_hidden[i][j].initialize_synaptic_weight([0.02 for k in range(network_width)])
		x.Neu_hidden[i][j].initialize_bias(0.02)
		
for i in range(target_var):
	x.Neu_output[i].initialize_synaptic_weight([0.02 for k in range(network_width)])
	x.Neu_output[i].initialize_bias(0.02)
"""initialize network setting"""

x.receive(input_signal,target_output)	#load input stream and target output into the network

x.feed_forward()
