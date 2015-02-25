from Class_Definition_ANN import ANN

training_example=[0,0,0]

print("Please initialize the state of your neurons, the synaptic weights and the network biases before continue")

print("Program now initialize the network with default setting")

network_depth=3			#set the depth of this network

network_width=3

x=ANN(network_depth,network_width)

for i in range(network_depth):
	for j in range(network_width):
		x.Neu[i][j].initialize_synaptic_weight([0.02 for k in range(network_width)])
		x.Neu[i][j].initialize_bias(0.02)
		
x.receive(training_example)	#load input stream into the network


"""
while not converged:
	forward()
	backward()
"""
