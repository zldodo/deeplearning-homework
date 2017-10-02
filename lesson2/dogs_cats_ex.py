#1 下载数据集， kaggle download
#2 整理数据文件
%pwd
import os,sys
current_dir=os.getcwd()
Lesson_home_dir=current_dir
data_home_dir=current_dir+'data/redux'
#没懂这句话，allow relatives imports to directories above lesson1
sys.path.insert(1,os.path.join(sys.path[0].'..'))

from utils import *
from vgg16 import vgg16
%matplotlib inline

# Action plan
# step1: creat validation and sample sets
# step2: rearrange image files into their respective directories
# step3: finetune and train maodel
# step4: generate predictions
# step5: validate predictions
# step6：submit predictions to Kaggle

# step1: creat validation and sample sets
%cd $data_home_dir
%mkdir valid
%mkdir results
%mkdir -p sample/train
%mkdir -p sample/valid
%mkdir -p sample/results
%mkdir -p sample/test
%mkdir -p test/unkown

%cd $data_home_dir/train
g=glob('*.jpg')
shuf=np.random.permutation(g)
for i in range(2000): os.rename(shuf[i],data_home_dir+'valid'+shuf[i])

from shutil import copyfile
g=glob('*.jpg')
shuf=np.random.permutation(g)
for i in range(200): copyfile(shuf[i],data_home_dir+'/sample/train'+shuf[i])

%cd $data_home_dir/valid
g=glob('*.jpg')
shuf=np.random/permutation(g)
for i in range(50): copyfile(shuf[i],data_home_dir+'/sample/valid'+shuf[i])

# step2: rearrange image files into their respective directories
%cd $data_home_dir/sample/train
%mkdir cats
%mkdir dogs
%mv cat.*.jpg cats/
%mv dog.*.jpg dogs/

%cd $data_home_dir/sample/valid
%mkdir cats
%mkdir dogs
%mv cat.*.jpg cats/
%mv dog.*.jpg dogs/

%cd $data_home_dir/valid
%mkdir cats
%mkdir dogs
%mv cat.*.jpg cats/
%mv dog.*.jpg dogs/

cd $data_home_dir/train
%mkdir cats
%mkdir dogs
%mv cat.*.jpg cats/
%mv dog.*.jpg dogs/

%cd $data_home_dir/test
%mv *.jpg unkown/

# step3: finetune and train maodel
%cd data_home_dir

path=data_home_dir+'/sample/'#'/'
test_path=data_home_dir+'/test/'
results_path=data_home_dir+'/results/'
train_path=path+'/train/'
valid_path=path+'/valid/'

vgg=Vgg16()
batch_size=64
no_of_epochs=3

#finetune the model
batches=vgg.get_batches(train_path,batch_size=batch_size)
val_batches=vgg.get_batches(valid_path,batch_size=batch_size)
vgg.finetune(batches)

vgg.model.optimizer.lr=0.01

last_weigths_filename=None
for epoch in range(no_of_epochs):
	print "Running epoch:%d" % epoch
	vgg.fit(batches,val_batches,nb_epoch=1)
	last_weigths_filename='ft%d.h5' % epoch
	vgg.model.save_weights(results_path+last_weigths_filename)
print "Completed %s fit operation" % no_of_epochs

# step4: generate predictions
batches,preds= vgg.test(test_path,batch_size=batch_size*2)
filenames=batches.filenames

save_array(results_path+'test_preds.dat',preds)
save_array(results_path+'filenames.dat',filenames)

# step5: validate predictions
vgg.model.load_weights(results_path+last_weigths_filename)
val_batches,probs=vgg.test(valid_path,batch_size=batch_size)

filenames=val_batches
expected_labels=val_batches.classes

our_predictions=probs[:,0]
our_labels=np.round(1-our_predictions)

from keras.preprocessing import image
def plot_idx(idx,titles=None)
	plots([image.load_img(valid_path+filenames[i]) for i in idx], titles=titles)

n_view=4
 #1 a few correct labels at random
correct=np.where(our_labels==expected_labels)[0]
idx= permutation(correct)[:n_view]
plots_idx(idx,our_predictions[idx])

#2 a few incorrect labels at random
incorrect=np.where(our_labelsr!=expected_labels)[0]
idx=permutation(incorrect)[:n_view]
plots_idx(idx,our_predictions[idx])

#3a the images we most confident were cats, and are actually cats
correct_cats=np.where((our_labels==0)&(our_labels==expected_labels))[0]
most_correct_cats=np.argsort(our_predictions[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats],our_predictions[correct_cats][most_correct_cats])

#3b as above, but dogs
correct_dogs=np.where((our_labels==1)&(our_labels==expected_labels))[0]
most_correct_dogs=np.argsort(our_predictions[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs],our_predictions[correct_dogs][most_correct_dogs])

#4a The image we were most confident were cats, but are actually dogs
incorrect_cats=np.where((our_labels==0)&&(our_labels!=expected_labels))[0]
most_incorrect_cats=np.argsort(our_predictions[incorrect_cats])[::-1][:n_view]
if len(most_incorrect_cats):
	plots_idx(incorrect_cats[most_incorrect_cats],our_predictions[incorrect_cats][most_incorrect_cats])
else:
	print('No incorrect cats！')

#4b The image we were most confident were dogs, but are actually cats
incorrect_dogs=np.where((our_labels==1)&&(our_labels!=expected_labels[:,1]))[0]
most_incorrect_dogs=np.argsort(our_predictions[incorrect_dogs])[:n_view]
if len(most_incorrect_cats):
	plots_idx(incorrect_dogs[most_incorrect_dogs],our_predictions[incorrect_dogs][most_incorrect_dogs])
else:
	print('No incorrect dogs！')

#5 The most uncertain labels(ie those with probability closest to 0.5)
most_uncertain=np.argsort(np.abs(our_predictions-0.5))
plots_idx(most_uncertain,our_predictions[most_uncertain])

from sklearn.metrics import confusion_matrix
cm=confusion_matrixn(expected_labels,our_labels)

plots_confusion_matrix(cm,val_batches.class_indices)


# step6：submit predictions to Kaggle
preds=load_array(results_path+'test_preds.dat')
filenames=load_array(results_path+'filenames.dat')

isdog=preds[:,1]
#Swap all ones with .95 and all zeros with 0.05
isdog=isdog.clip(min=0.05,max=0.95)

ids=np.array([int (f[8:f.find('.')]) for f in filenames])

subm=np.stack([ids,isdog],axis=1)

cd $data_home_dir
submission_file_name='submission1.csv'
np.savetxt(submission_file_name,subm,fmt='%d,%.5f',header='id,label',comments='')
 