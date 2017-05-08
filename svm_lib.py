import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import os

class svm(object):
	def __init__(self,data,index,feature_num,iteration=1000,batch_size=8,print_step=100,save_step=1000):
		self.data=data
		self.label=index #saved for multi-class classifiction
		self.index=index
		self.feature_num=feature_num
		self.iteration=iteration
		self.batch_size=batch_size
		self.print_step=print_step
		self.save_step=save_step
		self.saver=None

	def linear_svm_bi_class(self):
		x=tf.placeholder(tf.float,[None,self.feature_num])
		y_label=tf.placeholder(tf.float,[None,1])
		A=tf.Variable(tf.random_normal([self.feature_num,1]))
		b=tf.Variable(tf.random_normal([1,self.batch_size]))
		out=tf.matmul(x,A)+b
		norm=tf.reduce_sum(tf.square(A))
		res=1.-out*y_label
		res_m=tf.maximum(0.,res)
		classification=tf.reduce_mean(res_m)
		loss=norm+classification
		train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
		self.saver = tf.train.Saver()
		prediction = tf.sign(out)
		return train,loss,x,y_label,prediction

	def gaussian_svm_bi_class(self,gamma_val=-50.0):
		x=tf.placeholder('float',shape=[None,self.feature_num])
		prediction_grid = tf.placeholder('float',shape=[None,self.feature_num])
		y_label=tf.placeholder('float',[None,1])
		b=tf.Variable(tf.random_normal(shape=[1,self.batch_size]))
		gamma=tf.constant(gamma_val)
		dist = tf.reduce_sum(tf.square(x), 1)
		dist = tf.reshape(dist, [-1,1])
		sq_dists = tf.add(tf.sub(dist, tf.mul(2., tf.matmul(x, tf.transpose(x)))), tf.transpose(dist))
		my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))
		first_term = tf.reduce_sum(b)
		b_vec_cross = tf.matmul(tf.transpose(b), b)
		y_target_cross = tf.matmul(y_label, tf.transpose(y_label))
		second_term = tf.reduce_sum(tf.mul(my_kernel, tf.mul(b_vec_cross, y_target_cross)))
		loss = tf.neg(tf.sub(first_term, second_term))
		train = tf.train.GradientDescentOptimizer(0.002).minimize(loss)
		self.saver = tf.train.Saver()

		rA = tf.reshape(tf.reduce_sum(tf.square(x), 1),[-1,1])
		rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
		pred_sq_dist = tf.add(tf.sub(rA, tf.mul(2., tf.matmul(x, tf.transpose(prediction_grid)))), tf.transpose(rB))
		pred_kernel = tf.exp(tf.mul(gamma, tf.abs(pred_sq_dist)))
		prediction_output = tf.matmul(tf.mul(tf.transpose(y_label),b), pred_kernel)
		prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))

		return train,loss,x,y_label,prediction

	def sigmod_svm_bi_class(self,gamma_val=1,r=1):
		x=tf.placeholder('float',shape=[None,self.feature_num])
		y_label=tf.placeholder('float',[None,1])
		b=tf.Variable(tf.random_normal(shape=[1,self.batch_size]))
		gamma=tf.constant(gamma_val)
		r=tf.constant(r)
		x_square=tf.matmul(x, tf.transpose(x))
		my_kernel = tf.nn.tanh(tf.add(tf.matmul(gamma,x_square),r))
		first_term = tf.reduce_sum(b)
		b_vec_cross = tf.matmul(tf.transpose(b), b)
		y_target_cross = tf.matmul(y_label, tf.transpose(y_label))
		second_term = tf.reduce_sum(tf.mul(my_kernel, tf.mul(b_vec_cross, y_target_cross)))
		loss = tf.neg(tf.sub(first_term, second_term))
		train = tf.train.GradientDescentOptimizer(0.002).minimize(loss)
		self.saver = tf.train.Saver()
		return train,loss,x,y_label,prediction

	def linear_svr(self):
		x=tf.placeholder(tf.float,[None,self.feature_num])
		y_label=tf.placeholder(tf.float,[None,1])
		A=tf.Variable(tf.random_normal([self.feature_num,1]))
		b=tf.Variable(tf.random_normal([1,self.batch_size]))
		out=tf.matmul(x,A)+b
		norm=tf.reduce_sum(tf.square(A))
		res=1.-out*y_label
		res_m=tf.maximum(0.,res)
		classification=tf.reduce_mean(res_m)
		loss=norm+classification
		train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
		self.saver = tf.train.Saver()
		prediction=None
		return train,loss,x,y_label,prediction

	def train(self,sess,train,loss,x,y_label,name):
		for k in range(self.iteration):
			index=np.random.choice(len(self.data),size=self.batch_size)
			sel_x=[self.data[i] for i in index]
			sel_y=[[self.index[i]] for i in index]
			#print sel_x,sel_y
			sess.run(train,feed_dict={x:sel_x,y_label:sel_y})
			if k%self.print_step is 0:
				self.log_print(k,sess,loss,sel_x,sel_y)
			if k%self.save_step is 0:
				self.save(name,k)

	def log_print(self,step,sess,loss,sel_x,sel_y):
		print("STEP:",step,"   Loss is :", sess.run(loss,feed_dict={x:sel_x, y_label:sel_y}))
	
	def save(self,checkpoint_dir,step):
		model_name ="saver"
		model_dir = "%s" % ("save_model")
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(sess,
	                    os.path.join(checkpoint_dir, model_name),
	                    global_step=step)


	def load(checkpoint_dir,saver,sess):
	    print " [*] Reading checkpoints..."
	    
	    model_dir = "%s" % ("save_model")
	    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	    
	    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	    
	    if ckpt and ckpt.model_checkpoint_path:
	        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
	        print " [*] Success to read {}".format(ckpt_name)
	    else:
	        print "Failed"


	def init(self):
		sess=tf.Session()
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		return sess

	def train_one_model(self,select,num=''):
		if select is "linear":
			train,loss,x,y_label,prediction=svm.linear_svm_bi_class()
			sess=svm.init()
			svm.train(sess,train,loss,x,y_label,"linear_svm_bi_class"+num)
		if select is "gaussian":
			train,loss,x,y_label,prediction=svm.gaussian_svm_bi_class()
			sess=svm.init()
			svm.train(sess,train,loss,x,y_label,"gaussian_svm_bi_class"+num)
		if select is "sigmod":
			train,loss,x,y_label,prediction=svm.sigmod_svm_bi_class()
			sess=svm.init()
			svm.train(sess,train,loss,x,y_label,"sigmod_svm_bi_class"+num)
		if select is "linear_svr":
			train,loss,x,y_label,prediction=svm.linear_svr
			sess=svm.init()
			svm.train(sess,train,loss,x,y_label,"linear_svr"+num)


	def multi_class(self,select):
		#index should be preprocess as one hot like label=[[1,0,0],[0,1,0]]
		for a in range(len(self.label[0])):
			self.index=[]
			for label in self.label:
				if label[a] is 1:
					self.index.append(1)
				else:
					self.index.append(0)

			train_one_model(select,num=str(a))




class generate_psudo_data(object):
	def __init__(self):
		pass

	def generate_liner(self):
		iris = datasets.load_iris()
		x = np.array([[x[0], x[3]] for x in iris.data])
		y = np.array([1 if y==0 else -1 for y in iris.target])
		return x,y
	def generate_cycle(self):
		(x, y) = datasets.make_circles(n_samples=350, factor=.5, noise=.1)
		return x,y

x,y=generate_psudo_data().generate_cycle()
featurn_num=2
svm=svm(x,y,featurn_num)
train,loss,x,y_label,_=svm.gaussian_svm_bi_class() #if you have multiple models to train, build your model like this 
sess=svm.init()
svm.train(sess,train,loss,x,y_label,"gaussian_svm_bi_class")
# the file you have trained will be saved at the dir you spec


