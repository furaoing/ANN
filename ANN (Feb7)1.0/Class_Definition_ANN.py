#this py file define the class: ANN(Artificial Neuron Network)
from Class_Definition_Neuron import neuron

class ANN:
	"""define the ANN class. The number of layers per ANN network is predefined in variable layer_num, the number of neurons per layer is predefined in variable layer_volume"""
	layer_num=3      
	layer_volume=4
	
	N=list()   #Neuron object container, (layer_num)X(layer_volume) array
			
	for i in range(layer_num):
		N.append([neuron(i,k,layer_volume) for k in range(layer_volume)])	

	def receive(self,input):
		self.input=input
		
	def feed_forward(self):
		for 
