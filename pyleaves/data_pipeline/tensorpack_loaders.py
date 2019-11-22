
import numpy as np
import sys
from tensorpack.dataflow import DataFlow, MultiProcessRunner, BatchData
import threading


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return self.it.__next__()

def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g



class FilePathFlow(DataFlow):
	def __init__(self,paths,labels,size=(299,299)):
		self.paths = paths
		self.labels = labels
		self.size = size
	@threadsafe_generator
	def __iter__(self):
		return self.__next__()
	def __next__(self):
		for path,label in zip(self.paths,self.labels):
			yield [path, label]


#df = FilePathFlow(np.arange(0,1000),np.arange(0,1000))




#     def __iter__(self):
#         for p,label in zip(self.paths,self.labels):
#             try:
#                 image = cv2.resize(cv2.imread(p),self.size)
#                 rn = random.randint(1,200)
#                 if rn%21==0:
#                     cv2.imwrite('randomimg.jpg',image)
#                 yield image,label
#             except:
#                 print('problem with image %s'%p)
#                 continue

#sys.exec_info()[0]

def get_multiprocess_dataflow(paths, labels, size=(299,299), batch_size=32, num_prefetch=5, num_proc=10):
	ds = Dataflow(paths,labels,size=size)
	dsm = MultiProcessRunner(ds,num_prefetch=num_prefetch, num_proc=num_proc)
	ds1 = BatchData(dsm, batch_size)
	return ds1
# 	train_gen = gen(ds1)
# 	return train_gen


# img_size = (229,229)
# batch_size = 5
# ds = Dataflow(X_train,y_train,size=img_size)
# dsm = MultiProcessRunner(ds,num_prefetch=25, num_proc=10)
# ds1 = BatchData(dsm, batch_size)
# train_gen = gen(ds1)
# history= model.fit_generator(train_gen,callbacks=
#             callbacks_list, epochs=epochs,steps_per_epoch=len(y_train)/batch_size)