#this py file define the class: neuron 
import math

class neuron:
	"""Define the artificial neuron class. The number of hidden layers per ANN network is predefined in variable layer"""
	
	def __init__(self,layer,sn):
		self.layer=layer
		self.sn=sn
		self.sum=None
		self.out_signal=None
		
	def initialize_synaptic_weight(self,weight):
		self.weight=list()
		for i in range(len(weight)):
			self.weight.append(weight[i])
		
	def receive_weighted_and_addup(self,in_signal):
		"""Act as the adder in an artificial neuron"""
		#self.in_signal=in_signal
		self.sum=sum([self.weight[i]*in_signal[i] for i in range(len(in_signal))])
		
	def initialize_bias(self,bias):
		self.bias=bias
		
	def add_bias(self):
		self.sum=self.sum+self.bias
			
	def activate(self):
		self.out_signal=1/(1+math.exp(-self.sum))
			
	def backpropagation(self,error):
		self.error=error
