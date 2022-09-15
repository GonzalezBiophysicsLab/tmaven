import numpy as np
import struct

def read_spartan(filename):
	''' Following Appendix B.2 from https://www.scottcblanchardlab.com/_files/ugd/11765c_af3ab1af75474fd49d8350a4a98e601b.pdf
	* Ignoring endian-ness b/c not specified in format....
	'''
	with open(filename,'rb') as f:
		## Read and parse the information in the header
		buffer = f.read(22)
		header = struct.unpack('I4sHBBIIH',buffer)
		zero, signature, version, dataType, nChannels, nTraces, nFrames, chNameLen = header

		## Check if it is a Spartan file
		sig = signature.decode('utf8')
		if sig!='TRCS':
			raise Exception('Not a Spartan File. Signature is: %s'%(sig))

		## Read and parse the channel names
		buffer = f.read(chNameLen+nChannels-1) ## N-1 delimiters (none on the last one...)
		chNames = [b.replace(b'\x00',b' ').decode('utf8') for b in buffer.split(b'\x1f')]

		## Figure out numpy equivalent of dataType
		dtype = 'float32' if dataType == 9 else 'float64'

		## Read time data
		buffer = f.read(nFrames*4)
		time = np.frombuffer(buffer,dtype=dtype)

		## Read channels data
		channels = np.empty((nChannels,nTraces,nFrames),dtype=dtype)
		for ic in range(nChannels):
			buffer = f.read(nTraces*nFrames*4)
			channels[ic] = np.frombuffer(buffer,dtype=dtype).reshape((nFrames,nTraces)).T

		# print(channels.shape)
		channels = channels.astype('double')
		#### Read metadata
		## Note: an undocumented Null byte at the end of each metadataheader
		metadata = {}

		## read root metadata
		dataType,fieldSize,nameLen,fieldName,ndim,dataSize = read_metadatafield_header(f)

		## read root struct
		nFields, = struct.unpack('=B',f.read(1))
		for i in range(nFields): ## Loops over fileMetadata then traceMetadata
			nameLen_i,fieldName_i,isPacked_i = read_structfield_header(f)
			dataType_ii,fieldSize_ii,nameLen_ii,fieldName_ii,ndim_ii,dataSize_ii = read_metadatafield_header(f)
			metadata[fieldName_i] = {}

			nFields_i, = struct.unpack('=B',f.read(1))
			for j in range(nFields_i): ## Loops over the entires in this metadata
				nameLen_j,fieldName_j,isPacked_j = read_structfield_header(f)
				dataType_jj,fieldSize_jj,nameLen_jj,fieldName_jj,ndim_jj,dataSize_jj = read_metadatafield_header(f)

				## decode the contents
				buffer = f.read(fieldSize_jj-6-nameLen_jj-1)
				try:
					if dataType_jj in [0,12]:
						contents = buffer.decode('utf8')
					elif dataType_jj == 1:
						contents = np.frombuffer(buffer,dtype='uint8')
					elif dataType_jj == 2:
						contents = np.frombuffer(buffer,dtype='uint16')
					elif dataType_jj == 3:
						contents = np.frombuffer(buffer,dtype='uint32')
					elif dataType_jj == 4:
						contents = np.frombuffer(buffer,dtype='uint64')
					elif dataType_jj == 5:
						contents = np.frombuffer(buffer,dtype='int8')
					elif dataType_jj == 6:
						contents = np.frombuffer(buffer,dtype='int16')
					elif dataType_jj == 7:
						contents = np.frombuffer(buffer,dtype='int32')
					elif dataType_jj == 8:
						contents = np.frombuffer(buffer,dtype='int64')
					elif dataType_jj == 9:
						contents = np.frombuffer(buffer,dtype='float32')
					elif dataType_jj in [10]:
						if len(buffer)%8 != 0:
							buffer = buffer[3:] ### it's only for some and then it's always 3... wtf
						contents = np.frombuffer(buffer,dtype='float64')
					elif dataType_jj in [11,13]:
						raise Exception('WOAH') ## don't handle these
				except:
					contents = buffer

				metadata[fieldName_i][fieldName_j] = contents

	return chNames,time,channels,metadata


def read_metadatafield_header(f):
	## Note: fieldSize is size of everything from nameLen and on.... not contents.
	buffer = f.read(6)
	dataType,fieldSize,nameLen = struct.unpack('=BIB',buffer)
	fieldName = f.read(nameLen)#.decode('utf8') ## DON'T decode this! weird unwarranted utf8 errors appear
	buffer = f.read(5)
	ndim,dataSize = struct.unpack('=BI',buffer)
	f.read(1) ## null byte missing from documentation?!
	return dataType,fieldSize,nameLen,fieldName,ndim,dataSize

def read_structfield_header(f):
	nameLen, = struct.unpack('B',f.read(1))
	fieldName = f.read(nameLen).decode('utf8')
	isPacked, = struct.unpack('B',f.read(1))
	return nameLen,fieldName,isPacked

if __name__ == '__main__':
	chNames,time,channels,metadata = read_spartan('200ms.traces')
	import matplotlib.pyplot as plt
	plt.plot(channels[0,0,:])
	plt.plot(channels[1,0,:])
	plt.show()
	print(chNames)
