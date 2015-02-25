#this py file define the class: neuron 
import math

class neuron:
	"""Define the artificial neuron class. The number of layers per ANN network is predefined in variable layer_num, the number of neurons per layer is predefined in variable layer_volume"""
	
	def __init__(self,layer,sn,layer_volume):
		self.layer=layer
		self.sn=sn
		self.layer_volume=layer_volume  #number of neurons per layer
		self.sum=None
		self.out_signal=None
		
	def initialize_synaptic_weight(self,dendrite_num,weight):
		self.dendrite_num=dendrite_num
		self.weight=list()
		for i in range(dendrite_num):
			self.weight.append(weight[i])
		
	def receive_and_sum(self,in_signal):
		"""Act as the adder in an artificial neuron"""
		#self.in_signal=in_signal
		self.sum=sum([self.weight[i]*in_signal[i] for i in range(len(in_signal))])
			
	def activate(self):
		self.out_signal=1/(1+math.exp(-self.sum))
			
		
