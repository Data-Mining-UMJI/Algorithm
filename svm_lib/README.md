Usage:
The usage of svm_lib splits into the following steps 
1. Initialization & Setting
import this file. 
Then you are able to use the class svm(), use svm(arguments) to initial your svm models. 
The arguments includes: data,index,feature_num,iteration,batch_size,print_step,save_step.
Here are some arguments with initial values:
iteration=1000,batch_size=8,print_step=100,save_step=1000

2. Bulid the model
This library supports several classification methods with different kernel and support multiple classes.
Here are the list of them:
linear_svm_bi_class(self)  # linear svm classificating two classes
gaussian_svm_bi_class(self,gamma_val=-50.0)  # gaussian svm classificating two classes
sigmod_svm_bi_class(self,gamma_val=1,r=1)  # sigmod svm classificating two classes
linear_svr(self)  # linear svr, used for regression
multi_class(self,select)  # If you wants to have multi-class classification, run this first, it will do the pre-procession to turn it to bio class case.
save the return value: sess,train,loss,x,y_label, you will use them as input of training function.

3. Initialize the session
You could build several modes in step 2 at the same time, then before trainning, you should run:
sess=svm.init()

4.training:
the argument is like the following:
train(self,sess,train,loss,x,y_label,name)
Use the thing you have saved in step 2 as input.

5. Features
It also support operations like log print, save and load, you could run them by using the function:
log_print(self,step,sess,loss,sel_x,sel_y)
save(self,checkpoint_dir,step)
load(checkpoint_dir,saver,sess)

6. Example
Here is the example of the useage of this svm function:
x,y=generate_psudo_data().generate_cycle()
featurn_num=2
svm=svm(x,y,featurn_num)
train,loss,x,y_label,_=svm.gaussian_svm_bi_class() #if you have multiple models to train, build your model like this 
sess=svm.init()
svm.train(sess,train,loss,x,y_label,"gaussian_svm_bi_class")
