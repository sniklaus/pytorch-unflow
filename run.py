#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'css'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.size()) not in backwarp_tenGrid:
		tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
		tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Upconv(torch.nn.Module):
			def __init__(self):
				super(Upconv, self).__init__()

				self.netSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.netSixUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.netFivNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFivOut = torch.nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.netFivUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.netFouNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.netFouUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.netThrNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.netThrUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)

				self.netTwoNext = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)

				self.netUpscale = torch.nn.Sequential(
					torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False),
					torch.nn.ReplicationPad2d(padding=[ 0, 1, 0, 1 ])
				)
			# end

			def forward(self, tenFirst, tenSecond, objInput):
				objOutput = {}

				tenInput = objInput['conv6']
				objOutput['flow6'] = self.netSixOut(tenInput)
				tenInput = torch.cat([ objInput['conv5'], self.netFivNext(tenInput), self.netSixUp(objOutput['flow6']) ], 1)
				objOutput['flow5'] = self.netFivOut(tenInput)
				tenInput = torch.cat([ objInput['conv4'], self.netFouNext(tenInput), self.netFivUp(objOutput['flow5']) ], 1)
				objOutput['flow4'] = self.netFouOut(tenInput)
				tenInput = torch.cat([ objInput['conv3'], self.netThrNext(tenInput), self.netFouUp(objOutput['flow4']) ], 1)
				objOutput['flow3'] = self.netThrOut(tenInput)
				tenInput = torch.cat([ objInput['conv2'], self.netTwoNext(tenInput), self.netThrUp(objOutput['flow3']) ], 1)
				objOutput['flow2'] = self.netTwoOut(tenInput)

				return self.netUpscale(self.netUpscale(objOutput['flow2'])) * 20.0
			# end
		# end

		class Complex(torch.nn.Module):
			def __init__(self):
				super(Complex, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
					torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netRedir = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netCorrelation = correlation.ModuleCorrelation()

				self.netCombined = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				
				self.netFiv = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				
				self.netSix = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netUpconv = Upconv()
			# end

			def forward(self, tenFirst, tenSecond, tenFlow):
				objOutput = {}

				assert(tenFlow is None)

				objOutput['conv1'] = self.netOne(tenFirst)
				objOutput['conv2'] = self.netTwo(objOutput['conv1'])
				objOutput['conv3'] = self.netThr(objOutput['conv2'])

				tenRedir = self.netRedir(objOutput['conv3'])
				tenOther = self.netThr(self.netTwo(self.netOne(tenSecond)))
				tenCorr = self.netCorrelation(objOutput['conv3'], tenOther)

				objOutput['conv3'] = self.netCombined(torch.cat([ tenRedir, tenCorr ], 1))
				objOutput['conv4'] = self.netFou(objOutput['conv3'])
				objOutput['conv5'] = self.netFiv(objOutput['conv4'])
				objOutput['conv6'] = self.netSix(objOutput['conv5'])

				return self.netUpconv(tenFirst, tenSecond, objOutput)
			# end
		# end

		class Simple(torch.nn.Module):
			def __init__(self):
				super(Simple, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 2, 4, 2, 4 ]),
					torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 1, 3, 1, 3 ]),
					torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.ZeroPad2d([ 0, 2, 0, 2 ]),
					torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netUpconv = Upconv()
			# end

			def forward(self, tenFirst, tenSecond, tenFlow):
				objOutput = {}

				tenWarp = backwarp(tenInput=tenSecond, tenFlow=tenFlow)

				objOutput['conv1'] = self.netOne(torch.cat([ tenFirst, tenSecond, tenFlow, tenWarp, (tenFirst - tenWarp).abs() ], 1))
				objOutput['conv2'] = self.netTwo(objOutput['conv1'])
				objOutput['conv3'] = self.netThr(objOutput['conv2'])
				objOutput['conv4'] = self.netFou(objOutput['conv3'])
				objOutput['conv5'] = self.netFiv(objOutput['conv4'])
				objOutput['conv6'] = self.netSix(objOutput['conv5'])

				return self.netUpconv(tenFirst, tenSecond, objOutput)
			# end
		# end

		self.netFlownets = torch.nn.ModuleList([
			Complex(),
			Simple(),
			Simple()
		])

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(__file__.replace('run.py', 'network-' + arguments_strModel + '.pytorch')).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenFirst = tenFirst[:, [ 2, 1, 0 ], :, :]
		tenSecond = tenSecond[:, [ 2, 1, 0 ], :, :]

		tenFirst[:, 0, :, :] = tenFirst[:, 0, :, :] - (104.920005 / 255.0)
		tenFirst[:, 1, :, :] = tenFirst[:, 1, :, :] - (110.175300 / 255.0)
		tenFirst[:, 2, :, :] = tenFirst[:, 2, :, :] - (114.785955 / 255.0)

		tenSecond[:, 0, :, :] = tenSecond[:, 0, :, :] - (104.920005 / 255.0)
		tenSecond[:, 1, :, :] = tenSecond[:, 1, :, :] - (110.175300 / 255.0)
		tenSecond[:, 2, :, :] = tenSecond[:, 2, :, :] - (114.785955 / 255.0)

		tenFlow = None

		for netFlownet in self.netFlownets:
			tenFlow = netFlownet(tenFirst, tenSecond, tenFlow)
		# end

		return tenFlow
	# end
# end

netNetwork = None

##########################################################

def estimate(tenFirst, tenSecond):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().cuda().eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	assert(intWidth == 1280) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 384) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
	tenFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	tenSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

	tenOutput = estimate(tenFirst, tenSecond)

	objOutput = open(arguments_strOut, 'wb')

	numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
	numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
	numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

	objOutput.close()
# end