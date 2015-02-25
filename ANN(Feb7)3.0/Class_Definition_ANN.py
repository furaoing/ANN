#this py file define the class: ANN(Artificial Neu_hiddenron Network)
from Class_Definition_Neuron import neuron

class ANN:
	"""define the ANN class. The number of hidden layers per ANN network is defined in variable network_depth, the number of neurons per layer is defined in variable network_width"""
	def __init__(self,network_depth,network_width,target_var):
		self.network_depth=network_depth      
		self.network_width=network_width
		self.target_var=target_var
	
		self.Neu_hidden=list()   #Hidden neuron object container,(network_depth)X(network_width) array
			
		for i in range(self.network_depth): #initialize hidden layers
			self.Neu_hidden.append([neuron(i,k) for k in range(self.network_width)])
			
		self.Neu_output=list()   #Output neuron object container
			
		for i in range(self.target_var):    #initialize output layers
			self.Neu_output.append(neuron(-1,i))

	def receive(self,network_input,target_output):
		self.network_input=network_input
		self.target_output=target_output
		
	def feed_forward(self):
		self.network_output=[None for j in range(self.target_var)]
		self.error=None
		
		for j in range(self.network_width):  #first hidden layer
			self.Neu_hidden[0][j].receive_weighted_and_addup(self.network_input)
			self.Neu_hidden[0][j].add_bias()
			self.Neu_hidden[0][j].activate()
		
		for i in range(1,self.network_depth): #remaining hidden layer
			hiddenlayer_input=[self.Neu_hidden[i-1][k].out_signal for k in range(self.network_width)]
			
			for j in range(self.network_width):
				self.Neu_hidden[i][j].receive_weighted_and_addup(hiddenlayer_input)
				self.Neu_hidden[i][j].add_bias()
				self.Neu_hidden[i][j].activate()
				
		for j in range(self.target_var):  #output layer
			hiddenlayer_input=[self.Neu_hidden[self.network_depth-1][k].out_signal for k in range(self.network_width)]
			
			self.Neu_output[j].receive_weighted_and_addup(hiddenlayer_input)
			self.Neu_output[j].add_bias()
			self.Neu_output[j].activate()
		
			self.network_output[j]=self.Neu_output[j].out_signal
		
		self.error=0.5*sum([(self.network_output[j]-self.target_output[j])**2 for j in range(self.target_var)])