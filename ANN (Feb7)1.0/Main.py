from Class_Definition_ANN import ANN

feature=[0,0,0,0]
input=feature.append(1)  #add a bias signal into the input stream

x=ANN()

x.receive(input)	#load input stream into the network
	
"""
while not converged:
	forward()
	backward()
"""