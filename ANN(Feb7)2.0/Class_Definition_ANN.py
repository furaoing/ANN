#this py file define the class: ANN(Artificial Neuron Network)
from Class_Definition_Neuron import neuron

class ANN:
	"""define the ANN class. The number of layers per ANN network is defined in variable network_depth, the number of neurons per layer is defined in variable network_width"""
	def __init__(self,network_depth,network_width):
		self.network_depth=network_depth      
		self.network_width=network_width
	
		self.Neu=list()   #Neuron object container,(network_depth)X(network_width) array
			
		for i in range(self.network_depth):
			self.Neu.append([neuron(i,k) for k in range(self.network_width)])	

	def receive(self,input):
		self.input=input
		
	def feed_forward(self):
		i=0
		for j in range(self.network_width):
			self.Neu[i][j].receive_weighted_and_addup(self.input)
			self.Neu[i][j].add_bias()
			self.Neu[i][j].activate()
		
		for i in range(1,self.network_depth):
			signal_vector=[self.Neu[i-1][k].out_signal for k in range(self.network_width)]
			for j in range(self.network_width):
				self.Neu[i][j].receive_weighted_and_addup(signal_vector)
				self.Neu[i][j].add_bias()
				self.Neu[i][j].activate()