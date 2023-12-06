#!~/miniforge3/envs/torch-ml/bin/python

# Import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

# Block Module:
class Block(Module):
	
	def __init__(self, inChannels, outChannels):
		super().__init__()
		
		# Store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	
	def forward(self, x):
		# Apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))
	
class Encoder(Module):
	
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# Store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	
	def forward(self, x):
		# Initialize an empty list to store the intermediate outputs
		blockOutputs = []
		
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		
		# Return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
	
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# Initialize the number of channels, upsampler blocks, and decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)for i in range(len(channels) - 1)]
		)
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
		)
	
	def forward(self, x, encFeatures):
		# Loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# Crop the current features from the encoder blocks,
			# Concatenate them with the current upsampled features, and
			# Pass the concatenated output through the current decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		
		# Return the final decoder output
		return x
	
	def crop(self, encFeatures, x):
		# Grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		
		# Return the cropped features
		return encFeatures
	
encChannels = config.ENC_CHANNELS_DEFAULT
decChannels = config.DEC_CHANNELS_DEFAULT
outSize = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)

class UNet(Module):
	
	def __init__(self, encChannels=encChannels, decChannels=decChannels, nbClasses=1, retainDim=True, outSize=outSize):
		super().__init__()
		
		# Initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
		
	def forward(self, x):
		# Grab the features from the encoder
		encFeatures = self.encoder(x)
		
		# Pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		
		# Pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(decFeatures)
		
		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		
		# Return the segmentation map
		return map
	