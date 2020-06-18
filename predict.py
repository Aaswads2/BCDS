import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import statistics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
fileop = dir_path +'/output.jpg' 
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
img1=image
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./BCDS-model/model.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess,'BCDS-model/model.ckpt')

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)


#a function to read the labels name from labels.txt
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

#printing with labels
results = np.squeeze(result)
top_k = results.argsort()[-2:][::-1]
labels = load_labels('./labels.txt')
#print(top_k)
for i in top_k:
	msg = "{0} ---> : {1:>6.1%}"
	print(msg.format(labels[i], results[i]))
if(labels[top_k[0]]=='Cancerous'):
	#print('if true')
	ret,thresh2 = cv2.threshold(img1,100,255,cv2.THRESH_BINARY)
	median = cv2.medianBlur(thresh2,5)
	#cv2.imwrite('C:\\Users\\MILIND MOON\\Desktop\\medianblur3.jpg',median)
	t = 50
	img = median
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	(t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
	#cv2.imwrite('C:\\Users\\MILIND MOON\\Desktop\\newgray.jpg',gray)
	#cv2.imwrite('C:\\Users\\MILIND MOON\\Desktop\\newblur.jpg',blur)
	#cv2.imwrite("C:\\Users\\MILIND MOON\\Desktop\\bin.jpg", binary)


	# find contours
	(_,contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# print table of contours and sizes
	print("Found %d objects." % len(contours))
	l=[]
	#print(average(len(contours[0])))
	for (i, c) in enumerate(contours):
		l.append(len(c))
	print(l)
	avg=sum(l)/len(l)
	print(avg)
	try:
		med=statistics.stdev(l)
	except statistics.StatisticsError:
		med=avg
	print(med)
	for (i, c) in enumerate(contours):
		print("\tSize of contour %d: %d" % (i, len(c)))
	print("after and")
	for (i, c) in enumerate(contours):
		if (len(c)>=med):
			cv2.drawContours(img1, c, -1, (0, 0, 255), 3)
			print("\tSize of contour %d: %d" % (i, len(c)))
	cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	cv2.imshow("output", img1)
	cv2.waitKey(0)

