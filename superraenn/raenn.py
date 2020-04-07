# lstm autoencoder recreate sequence
from keras.models import Model
from keras.layers import Input,GRU, TimeDistributed
from keras.layers import Dense,Concatenate,concatenate
from keras.layers import RepeatVector
from keras.utils import plot_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler,\
            EarlyStopping,TerminateOnNaN,ReduceLROnPlateau
import math
import datetime


def customLoss(yTrue,yPred):
    return K.mean(K.square(yTrue[:,:,1:5] - yPred[:,:,:]))

def prep_input(input_lcs):
    lens = []
    for thing in x_arr:
        my_thing = thing.x.astype('float64')
        lens.append(len(my_thing))
    sequence_len = np.max(lens)
    maxlen = np.max(lens)

    #Get rid of points with len<3
    gind = np.where(np.asarray(lens)>2)
    x_arr = x_arr[gind]
    lens = np.asarray(lens)[gind]
    ids = ids[gind]
    lms = lms[gind]
    N = len(x_arr)
    #convert from LC format to list of arrays
    sequence = np.zeros((len(x_arr),sequence_len,9))
    for i,thing in enumerate(x_arr):
        my_thing = thing.x.astype('float64')
        sequence[i,0:lens[i],0] = my_thing[:,0,1]
        sequence[i,0:lens[i],1:5] = my_thing[:,:,2]
        sequence[i,0:lens[i],5:] = my_thing[:,:,0]
        sequence[i,lens[i]:,0] = np.max(my_thing[:,0,1])+100.0
        sequence[i,lens[i]:,5:] = 1.0
    sequence[:,:,1:5] = -1.0 * sequence[:,:,1:5]
    bandmin = np.min(sequence[:,:,1:5])
    bandmax = np.max(sequence[:,:,1:5])
    sequence[:,:,1:5] = (sequence[:,:,1:5] - bandmin) / (bandmax - bandmin) 
    new_lms = np.reshape(np.repeat(lms,sequence_len),(len(lms),-1))
    outseq = np.reshape(sequence[:,:,0],(len(sequence),sequence_len,1)) * 1.0
    outseq = np.dstack((outseq,new_lms))
    return sequence, outseq, maxlen

#Scheduler
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def make_model(LSTMN, encodingN, maxlen):
    # define model
    input_1 = Input((None,9)) #layer 0
    input_2 = Input((maxlen,2))

    encoder1 = GRU(LSTMN,return_sequences=True,activation='tanh',recurrent_activation='hard_sigmoid')(input_1)#layer 1
    encoded = GRU(encodingN,return_sequences=False,activation='tanh',recurrent_activation='hard_sigmoid')(encoder1) #layer 2
    repeater = RepeatVector(maxlen)(encoded)
    merged = concatenate([repeater, input_2],axis=-1) #layer 3,4
    decoder1 = GRU(LSTMN,return_sequences=True,activation='tanh',recurrent_activation='hard_sigmoid')(merged)#layer 5
    decoder2 = TimeDistributed(Dense(4,activation='tanh'),
                input_shape=(None,1))(decoder1)#layer 6

    model = Model(input=[input_1, input_2], output=decoder2)

    for layer in model.layers:
        print(layer.output_shape)

    new_optimizer =Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, 
                        decay=0)
    model.compile(optimizer=new_optimizer, loss=customLoss)

    lrate = LearningRateScheduler(step_decay)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
                        verbose=0, mode='min', baseline=None, 
                        restore_best_weights=True)
    tnan = TerminateOnNaN()
    rlr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, 
                        verbose=0, mode='auto', min_delta=0.0001, 
                        cooldown=0, min_lr=0)
    callbacks_list = [es]
    return model, callbacks_list, input_1, encoded

def fit_model(model, callbacks_list, sequence, outseq, n_epoch):
    model.fit([sequence,outseq], sequence, epochs=n_epoch,  verbose=1, 
            shuffle=False,callbacks=callbacks_list,validation_split=0.33)
    return model

def test_model(sequence_test,model,lms, sequence_len,plot=True):
    outseq_test = np.reshape(sequence_test[:,:,0],(len(sequence_test),sequence_len,1))
    lms_test = np.reshape(np.repeat(lms[13],sequence_len),(len(sequence_test),-1))
    outseq_test = np.reshape(outseq_test[:,:,0],(len(sequence_test),sequence_len,1))
    outseq_test = np.dstack((outseq_test,lms_test))

    yhat = model.predict([sequence_test,outseq_test], verbose=1)
    print(yhat,np.shape(yhat),sequence_test,outseq_test)
    if plot:
        plt.plot(sequence_test[0,:,0],yhat[0,:,1],color='grey')
        plt.plot(sequence_test[0,:,0],sequence_test[0,:,2],color='grey')
        plt.show()
def get_encoder(model, input_1, encoded):
    encoder = Model(input=input_1, output=encoded)
    return encoder

def get_decoder(model,encodingN):
    encoded_input = Input(shape=(None,(encodingN+2)))
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    decoder = Model(input=encoded_input,output=decoder_layer3(decoder_layer2(encoded_input)))
    return decoder

def get_decodings(decoder,encoder,sequence,lms, encodingN, sequence_len,plot=True):
    seq = np.reshape(sequence[0,:,:],(1,sequence_len,9))
    encoding1 = encoder.predict(seq)[-1]
    encoding1 = np.vstack([encoding1]).reshape((1,1,encodingN))
    repeater1 = np.repeat(encoding1,sequence_len,axis=1)
    out_seq = np.reshape(seq[:,:,0],(len(seq),sequence_len,1))
    lms_test = np.reshape(np.repeat(lms[0],sequence_len),(len(seq),-1))
    out_seq = np.dstack((out_seq,lms_test))

    decoding_input2 = np.concatenate((repeater1,out_seq),axis=-1)

    decoding2 = decoder.predict(decoding_input2)[0]
    print(seq)

    if plot:
        plt.plot(seq[0,:,0],seq[0,:,1],'green',alpha=1.0,linewidth=1)
        plt.plot(seq[0,:,0],decoding2[:,0],'green',alpha=0.2,linewidth=10)
        plt.plot(seq[0,:,0],seq[0,:,2],'red',alpha=1.0,linewidth=1)
        plt.plot(seq[0,:,0],decoding2[:,1],'red',alpha=0.2,linewidth=10)
        plt.plot(seq[0,:,0],seq[0,:,3],'orange',alpha=1.0,linewidth=1)
        plt.plot(seq[0,:,0],decoding2[:,2],'orange',alpha=0.2,linewidth=10)
        plt.plot(seq[0,:,0],seq[0,:,4],'purple',alpha=1.0,linewidth=1)
        plt.plot(seq[0,:,0],decoding2[:,3],'purple',alpha=0.2,linewidth=10)
        plt.show()

def save_model(model, encodingN, LSTMN):
    # serialize model to JSON
    now = datetime.datetime.now()
    date = str(now.strftime("%Y-%m-%d"))
    model_json = model.to_json()
    with open("./models/model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./models/model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".h5")
    print("Saved model to disk")

def save_encodings(model,encoder, sequence,ids, INPUT_FILE, encodingN,LSTMN,N, sequence_len):
    now = datetime.datetime.now()
    date = str(now.strftime("%Y-%m-%d"))
    encodings = np.zeros((N,encodingN))
    for i in np.arange(N):
        seq = np.reshape(sequence[i,:,:],(1,sequence_len,9))

        my_encoding = encoder.predict(seq)
        print(my_encoding,np.shape(my_encoding))

        encodings[i,:] = my_encoding
        encoder.reset_states() 


    encoder_sne_file = './encodings/en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'.npz'
    np.savez(encoder_sne_file,encodings=encodings,ids = ids,INPUT_FILE=INPUT_FILE)

