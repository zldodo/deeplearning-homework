%matplotlib inline
import utils;reload(utils)
from utils import *

# creat Dense() layers, linear model, followed by a simple activation function

path = "data/redux/"
model_path=path+'models/'
if not os.path.exsits(model_path): os.mikdir(model_path)

batch_size=100

from vgg16 import Vgg16
vgg=Vgg16()
model=vgg.model

val_batches=get_batches(path+'valid',shuffle=False,batch_size=1)
batches=get_batches(path+'train',shuffle=False,batch_size=1)

import bcolz
def save_array(fname,arr):c=bcolz.carry(arr, rootdir=fname,mode='w');c.flush()
def load_array(fname):return bcolz.open(fname)[:]

val_data=get_data(path+'valid')
trn_data=get_data(path+'train')

save_array(model_path+'train_data.bc',trn_data)
save_array(model_path+'valid_data.bc',val_data)

trn_data=load_array(model_path+'train_data.bc')
val_data=load_array(model_path+'valid_data.bc')
#check trn_data.shape

def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

val_classes=val_batches.classes
trn_classes=batches.classes
#check trn_classes.shape
val_labels=onehot(val_classes)
trn_labels=onehot(trn_classes)
#check trn_labels.shape

trn_features=model.predict(trn.data, batch_size=batch_size)
val_features=model.predict(val.data,batch_size=batch_size)

save_array(model_path+'trn_lastlayer_features.bc',trn_features)
save_array(model_path+'val_lastlayer_features.bc',val_features)

trn_features=load_array(model_path+'trn_lastlayer_features.bc')
val_features=load_array(model_path+'val_lastlayer_features.bc')

#define linear model
lm=Sequential([Dense(2,activation='softmax',input_shape=(1000,))])
lm.complie(optimizer=RMSprop(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])


#fit the model
batch_size=64
lm.fit(trn_features,trn_labels,nb_epoch=3,batch_size=batch_size,
	validation_data=(val_data,val_labels))

#save weights
model.save_weights(model_path+'finetune1.h5')
model.load_weights(model_path+'finetune1.h5')
