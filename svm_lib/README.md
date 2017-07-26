Usage:
====

The usage of svm_lib splits into the following steps <br> 

1. Initialization & Setting<br> 
==
import this file. <br> 
Then you are able to use the class svm(), use svm(arguments) to initial your svm models. <br> 
The arguments includes: data,index,feature_num,iteration,batch_size,print_step,save_step.<br> 
Here are some arguments with initial values:<br> 
iteration=1000,batch_size=8,print_step=100,save_step=1000<br> 

#2. Bulid the model<br> 
==
This library supports several classification methods with different kernel and support multiple classes.<br> 
Here are the list of them:<br> 
linear_svm_bi_class(self)  # linear svm classificating two classes<br> 
gaussian_svm_bi_class(self,gamma_val=-50.0)  # gaussian svm classificating two classes<br> 
sigmod_svm_bi_class(self,gamma_val=1,r=1)  # sigmod svm classificating two classes<br> 
linear_svr(self)  # linear svr, used for regression<br> 
multi_class(self,select)  # If you wants to have multi-class classification, run this first, it will do the pre-procession to turn it to bio class case.<br> 
save the return value: sess,train,loss,x,y_label, you will use them as input of training function.<br> 

#3. Initialize the session<br> 
==
You could build several modes in step 2 at the same time, then before trainning, you should run:<br> 
sess=svm.init()<br> 

#4.training<br> 
==
the argument is like the following:<br> 
train(self,sess,train,loss,x,ylabel,name)<br> 
Use the thing you have saved in step 2 as input.<br> 

#5. Features<br> 
==
It also support operations like log print, save and load, you could run them by using the function:<br> 
log_print(self,step,sess,loss,sel_x,sel_y)<br> 
save(self,checkpoint_dir,step)<br> 
load(checkpoint_dir,saver,sess)<br> 

#6. Example<br> 
==
Here is the example of the useage of this svm function:<br> 
x,y=generate_psudo_data().generate_cycle()<br> 
featurn_num=2<br> 
svm=svm(x,y,featurn_num)<br> 
train,loss,x,y_label,_=svm.gaussian_svm_bi_class() #if you have multiple models to train, build your model like this <br> 
sess=svm.init() <br> 
svm.train(sess,train,loss,x,y_label,"gaussian_svm_bi_class")<br> 
