# Authors: PhD. Nguyen Hong Quang
# School of Information and Communication Technology
# Hanoi University of Science and Technology
# Email: quangnh@soict.hust.edu.vn
import numpy as np
import sys
import os
import pickle
import time
import math
import numpy as np
import glob
import random
import pdb
import csv
import torch 

 
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from keras.models import Sequential
import keras.layers.core as core
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization, Lambda, GlobalMaxPooling2D
from keras.layers import LSTM, Bidirectional, Reshape, Layer
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import merge, Input, TimeDistributed
from keras.regularizers import WeightRegularizer, l1, l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import class_weight
from keras import objectives, initializations
from keras import backend as K

############# HYPER-PARAMETERS ############
FILE_MODEL_TMP = "model_tmp.pkl"

MY_RANDOM_STATE = 5 
torch.manual_seed(MY_RANDOM_STATE)

SAMPLE_LENGTH = 200
AVGPOOL1D_KERNEL_SIZE = 4
CONV1D_KERNEL_SIZE = 3
CONV1D_FEATURE_SIZE_BLOCK1 = 32
CONV1D_FEATURE_SIZE_BLOCK2 = 64
CONV1D_FEATURE_SIZE_BLOCK3 = 128

FULLY_CONNECTED_LAYER_SIZE = 256

MODEL_DIR = '../Train/model_layer1_seed' + str(MY_RANDOM_STATE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
###########################################
my_dict = {'A': 0,
        'C': 1, 
        'G': 2,
        'T':3,
        'a':0,
        'c':1,
        'g':2,
        't':3}

# data = one_hot(1,3) ==> [0. 1. 0.]        
def one_hot(index, dimension):
    data = np.zeros((dimension))
    data[index] = 1
    return data

#data = one_hot(1,3)
#print(data)


def load_text_file(file_text):
    with open(file_text) as f:
        lines = f.readlines()
        my_data = [line.strip().upper() for line in lines[1::2]]
        return my_data

class EnhancerDataset(Dataset):
    # X: list of Enhancer sequences (200 characters for each sequence)
    # Y: list label [0, 1]; 0: negative, 1: positive
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        label = self.Y[index]
        sample = self.X[index]
        
        values = np.zeros((4, SAMPLE_LENGTH))
        for i in range(SAMPLE_LENGTH):
            char_idx = my_dict[sample[i]]
            values[char_idx, i] = 1 
        
        values_one_mer = self.extract_1_mer(sample)
        #values = np.concatenate((values, values_one_mer), axis=0)
        values_two_mer = self.extract_2_mer(sample)
        #values = np.concatenate((values, values_two_mer), axis=0)
        values_three_mer = self.extract_3_mer(sample)
        #values = np.concatenate((values, values_three_mer), axis=0)
        values = np.concatenate((values, values_one_mer, values_two_mer, 
                        values_three_mer), axis=0)
        
        input = torch.from_numpy(values)
        return input, label
    
    def extract_1_mer(self, sample):
        my_count = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}        
        values = np.zeros((1, SAMPLE_LENGTH))
        for i in range(SAMPLE_LENGTH):
            my_count[sample[i]] += 1
        
        #for one_mer in my_count:
        #    print("one mer: ", one_mer, " : ", my_count[one_mer])
        
        for i in range(SAMPLE_LENGTH):
            values[0, i] = my_count[sample[i]] / SAMPLE_LENGTH;
        
        #print("values: ", values)    
        return values
    
    def extract_2_mer(self, sample):
        my_count = {'AA': 0.0, 'AC': 0.0, 'AG': 0.0, 'AT': 0.0,
                    'CA': 0.0, 'CC': 0.0, 'CG': 0.0, 'CT': 0.0,
                    'GA': 0.0, 'GC': 0.0, 'GG': 0.0, 'GT': 0.0,
                    'TA': 0.0, 'TC': 0.0, 'TG': 0.0, 'TT': 0.0} 
        values = np.zeros((2, SAMPLE_LENGTH))
        for i in range(SAMPLE_LENGTH - 1):
            two_mer = sample[i:i+2]
            #print("two_mer: ", two_mer)
            my_count[two_mer] += 1
        
        #for two_mer in my_count:
        #    print("two mer: ", two_mer, " : ", my_count[two_mer])
        
        values = np.zeros((2, SAMPLE_LENGTH))
        for i in range(1,SAMPLE_LENGTH-1):
            two_mer_left = sample[i-1:i+1]
            two_mer_right = sample[i:i+2]
            
            values[0, i] = my_count[two_mer_left] / (SAMPLE_LENGTH - 1);
            values[1, i] = my_count[two_mer_right] / (SAMPLE_LENGTH - 1);
        
        #print("values: ", values) 
        return values
    
    def extract_3_mer(self, sample):
        my_count = {}
                                        
        for firchCh in ['A', 'C', 'G', 'T']:
            for secondCh in ['A', 'C', 'G', 'T']:
                for thirdCh in ['A', 'C', 'G', 'T']:
                    three_mer = firchCh + secondCh + thirdCh
                    my_count[three_mer] = 0.0
        for i in range(SAMPLE_LENGTH - 2):
            three_mer = sample[i:i+3]
            #print("two_mer: ", two_mer)
            my_count[three_mer] += 1
                    
        values = np.zeros((1, SAMPLE_LENGTH))
        for i in range(1,SAMPLE_LENGTH-2):
            three_mer = sample[i-1:i+2]
            values[0, i] = my_count[three_mer] / SAMPLE_LENGTH;
                    
        return values
        
    def __len__(self):
        #return 100
        return len(self.X)
        
class EnhancerCnnModel(nn.Module,input_dim=4, input_length=2705, nbfilter = 101):
    model = Sequential()
    # model.add(brnn)

    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=10,
                     border_mode="valid",
                     # activation="relu",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_length=3))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(nbfilter * 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    return model
    def __init__(self):
        super(EnhancerCnnModel, self).__init__()
        self.c1_1 = nn.Conv1d(8, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE, padding=1)
        self.c1_1bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.c1_2 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK1, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c1_2bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.c1_3 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK1, 
            CONV1D_KERNEL_SIZE, padding=1)    
        self.c1_3bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.p1 = nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)
        
        self.c2_1 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, 
            CONV1D_FEATURE_SIZE_BLOCK2, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c2_1bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.c2_2 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK2, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c2_2bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.c2_3 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK2, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c2_3bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.p2 = nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)
        
        self.fc = nn.Linear(768, FULLY_CONNECTED_LAYER_SIZE)        
        self.out = nn.Linear(FULLY_CONNECTED_LAYER_SIZE, 1)
        
        self.criterion = nn.BCELoss()        
     
    def forward(self, inputs):
        batch_size = inputs.size(0)
        # Turn (batch_size x seq_len) into (batch_size x input_size x seq_len) for CNN
        #inputs = inputs.transpose(1,2)
        #print("inputs size: ", inputs.size())        
        output = F.relu(self.c1_1bn(self.c1_1(inputs)))
        output = F.relu(self.c1_2bn(self.c1_2(output)))
        output = F.relu(self.c1_3bn(self.c1_3(output)))
        output = self.p1(output)
        #print("After p1: ", output.shape) 
        
        output = F.relu(self.c2_1bn(self.c2_1(output)))
        output = F.relu(self.c2_2bn(self.c2_2(output)))
        output = F.relu(self.c2_3bn(self.c2_3(output)))
        output = self.p2(output)
        #print("After p2: ", output.shape)
        
        output = output.view(batch_size, -1)
        #print("Reshape : ", output.shape)
        
        output = F.relu(self.fc(output))
        #print("After FC layer: ", output.shape)  
        
        output = torch.sigmoid(self.out(output))
        #print("Final output (After sigmoid): ", output.shape)
        #print("Final output: ", output)
        
        return output 
    
def evaluate(file_model, loader):
    #model.eval()
    model = EnhancerCnnModel()
    #print("CNN Model: ", model)
    if torch.cuda.is_available(): model.cuda()
    
    model.load_state_dict(torch.load(file_model))
    model.eval()    
    
    epoch_loss = 0.0
    nb_samples = 0
    
    arr_labels = []
    arr_labels_hyp = []
    arr_prob = []
    
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data
        #print("labels: ", labels)
        
        inputs_length = inputs.size()[0]
        nb_samples += inputs_length
        
        arr_labels += labels.squeeze(1).data.cpu().numpy().tolist()

        inputs = inputs.float()
        labels = labels.float()
        
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        
        outputs = model(inputs)
        loss = model.criterion(outputs, labels)
        
        epoch_loss = epoch_loss + loss.item() * inputs_length
        
        arr_prob += outputs.squeeze(1).data.cpu().numpy().tolist()
    
    print("nb_samples: ", nb_samples)
    epoch_loss_avg = epoch_loss / nb_samples
    print("epoch loss avg: ", epoch_loss_avg)
        
    print("arr_prob: ", len(arr_prob))
    print("arr_labels: ", len(arr_labels))
    
    auc = metrics.roc_auc_score(arr_labels, arr_prob)
    print("auc: ", auc)
    
    arr_labels_hyp = [int(prob > 0.5) for prob in arr_prob]
    #print("arr_prob: ", arr_prob)
    #print("arr_labels_hyp: ", arr_labels_hyp)
    arr_labels = [int(label) for label in arr_labels]
    
    return result
def get_bag_data_1_channel(seqs, labels, max_len = 2695):
    bags = []
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        #bag_subt.append(tri_fea.T)
        bags.append(np.array(tri_fea))
    return bags, labels
	
def get_domain_features(in_file = 'rbps_HT.txt'):
    protein_list = []
    with open('protein_list', 'r') as fp:
        for line in fp:
            protein_list.append(line[1:-1])
    domain_dict = {}
    fp = open(in_file, 'r')
    index = 0
    for line in fp:
        values = line.rstrip().split()
        vals = [float(val) for val in values]
        domain_dict[protein_list[index]] = vals
        index = index + 1
    fp.close()

    return domain_dict

def get_domain_features(in_file = 'rbps_HT.txt'):
    protein_list = []
    with open('protein_list', 'r') as fp:
        for line in fp:
            protein_list.append(line[1:-1])
    domain_dict = {}
    fp = open(in_file, 'r')
    index = 0
    for line in fp:
        values = line.rstrip().split()
        vals = [float(val) for val in values]
        domain_dict[protein_list[index]] = vals
        index = index + 1
    fp.close()

    return domain_dict
	
def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label  

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:] #discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict


def read_fasta_file_new(fasta_file = '../data/UTR_hg19.fasta'):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:].split()[0]  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    return seq_dict

def focal_loss(gamma=2, alpha=2):
	def focal_loss_fixed(y_true, y_pred):
		if(K.backend()=="tensorflow"):
			import tensorflow as tf
			pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
		if(K.backend()=="theano"):
			import theano.tensor as T
			pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
	return focal_loss_fixed

def perform_metrics(out, y_test):
    threshold = np.arange(0.1, 0.9, 0.1)

    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:, i])
        for j in threshold:
            y_pred = [1 if prob >= j else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(y_test[:, i], y_pred))
        acc = np.array(acc)
        index = np.where(acc == acc.max())
        accuracies.append(acc.max())
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    print "best thresholds", best_threshold
    y_pred = np.array(
        [[1 if out[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

    print("-" * 40)
    print("Matthews Correlation Coefficient")
    print("Class wise accuracies")
    print(accuracies)

    print("other statistics\n")
    total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i] == y_pred[i]).sum() == 5])
    print("Fully correct output")
    print(total_correctly_predicted)
    print(total_correctly_predicted / out.shape[0])


def load_rnacomend_data(datadir='../data/'):
    pair_file = datadir + 'interactions_HT.txt'
    # rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'

    rna_seq_dict = read_fasta_file(rna_seq_file)
    protein_set = set()
    inter_pair = {}
    new_pair = {}
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            protein_set.add(protein)
            rna = values[1]
            inter_pair.setdefault(rna, []).append(protein)
            new_pair.setdefault(protein, []).append(rna)

    for protein, rna in new_pair.iteritems():
        print protein, len(rna)
    return inter_pair, rna_seq_dict, protein_set


def get_rnarecommend(inter_pair_dict, rna_seq_dict, protein_list):
    data = {}
    labels = []
    rna_seqs = []
    protein_list.append("negative")
    all_hg19_utrs = read_fasta_file_new()
    remained_rnas = list(set(all_hg19_utrs.keys()) - set(inter_pair_dict.keys()))
    #pdb.set_trace()
    for rna, protein in inter_pair_dict.iteritems():
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        init_labels = np.array([0]*len(protein_list))
        inds = []
        for pro in protein:
            inds.append(protein_list.index(pro))
        init_labels[np.array(inds)] = 1
        labels.append(init_labels)
        rna_seqs.append(rna_seq)
    #pdb.set_trace()
    max_num_targets = np.sum(labels, axis =0).max()
    # negatives

    random.shuffle(remained_rnas)
    #pdb.set_trace()
    for rna in remained_rnas[:max_num_targets]:
        rna_seq = all_hg19_utrs[rna]
        rna_seq = rna_seq.replace('T', 'U')
        rna_seqs.append(rna_seq)
        init_labels = np.array([0] * (len(protein_list) - 1) + [1])
        labels.append(init_labels)

    data["seq"] = rna_seqs
    data["Y"] = np.array(labels)

    return data


def get_RNA_seq_concolutional_array(seq, motif_len = 10):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    # for seq in seqs:
    # for key, seq in seqs.iteritems():
    half_len = motif_len/2
    row = (len(seq) + 2 * half_len)
    new_array = np.zeros((row, 4))
    for i in range(half_len):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - half_len, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + half_len
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
            # data[key] = new_array
    return new_array

def padding_sequence(seq, max_len = 2695, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_class_weight(df_y):
    y_classes = df_y.idxmax(1, skipna=False)

    from sklearn.preprocessing import LabelEncoder

    # Instantiate the label encoder
    le = LabelEncoder()

    # Fit the label encoder to our label series
    le.fit(list(y_classes))

    # Create integer based labels Series
    y_integers = le.transform(list(y_classes))

    # Create dict of labels : integer representation
    labels_and_integers = dict(zip(y_classes, y_integers))

    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    sample_weights = compute_sample_weight('balanced', y_integers)

    class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))

    return class_weights_dict


def get_all_rna_mildata(seqs, labels, training_val_indice, train_val_label, test_indice, test_label, max_len = 2695):
    index = 0
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])
    train_bags, train_labels = get_bag_data_1_channel(train_seqs, train_val_label, max_len= max_len)

    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])

    test_bags, test_labels = get_bag_data_1_channel(test_seqs, test_label, max_len=max_len)

    return train_bags, train_labels, test_bags, test_labels

        
# dataset = {"data_test" : testset["data"], "label_test" : testset["label"]}


def read_test_file(filename):
    text_file = open(filename)
    lines = text_file.readlines()
    m = len(lines)//5
    my_data = []
    for i in range(m):
        text = lines[i*5+1].strip() + lines[i*5+2].strip() + \
            lines[i*5+3].strip() + lines[i*5+4].strip()
        my_data.append(text.upper())
    
    return my_data

def prepare_test_data():
    print("\n ==> Loading test set")
    data_strong = read_test_file('test_strong_enhancer.txt')
    print("data_strong: ", len(data_strong))
    #print("data_strong: ", data_strong)
    
    data_weak = read_test_file('test_weak_enhancer.txt')
    print("data_weak: ", len(data_weak))
    
    data_enhancer = data_strong + data_weak
    print("data_enhancer: ", len(data_enhancer))
    
    data_non_enhancer = read_test_file('test_non_enhancer.txt')
    print("data_non_enhancer: ", len(data_non_enhancer))
    
    label_enhancer = np.ones((len(data_enhancer),1))
    label_non_enhancer = np.zeros((len(data_non_enhancer), 1))
    
    data = np.concatenate((data_enhancer, data_non_enhancer))
    label = np.concatenate((label_enhancer, label_non_enhancer))
    
    testset = {"data" : data, "label" : label}
    
    return testset

def testing():
    test_dataset = EnhancerDataset(testset["data"], testset["label"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=32,                              
                              shuffle=False, num_workers=4)
          
    with open(testresult_fn, mode='w') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
       
    list_model_fn = sorted(glob.glob(MODEL_DIR+"/enhancer_*.pkl"))
    #print(list_model_fn)
    y_prob_mtx = []
    
    for model_fn in list_model_fn:
        print(model_fn)
        result = evaluate(model_fn, test_loader)
        #print(result['arr_prob'])
        y_prob_mtx.append(result['arr_prob'])
        #break
        
        with open(testresult_fn, mode='a') as outfile:
            outfile = csv.writer(outfile, delimiter=',')
            outfile.writerow([model_fn, result['acc'], result['auc'], 
			roc_auc_score(test_labels, preds, average='macro')
			roc_auc_score(test_labels, preds, average='micro'
			roc_auc_score(test_labels, preds, average='weighted'
            print('ACC:', metrics.accuracy_score(test_labels, preds))
			confusion_matrix(test_labels, preds,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
			                                            ,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]))
            #confusion_matrix output 
			metrics.classification_report(y_test, y_predict,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
			                                            ,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])))
            #classification_report output 
			#print('Macro-ACC:',metrics.precision_score(y_test,y_predict,average='macro')) 
            # print('Micro-ACC:', metrics.precision_score(y_test, y_predict, average='micro'))
            # print('Weighted-ACC :', metrics.precision_score(y_test, y_predict, average='weighted'))
 
 
    print 'Macro-AUC', roc_auc_score(test_labels, preds, average='macro')
    print 'Micro-AUC',roc_auc_score(test_labels, preds, average='micro')
    print 'weight-AUC',roc_auc_score(test_labels, preds, average='weighted')
  

    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0

    print f1_score(test_labels, preds, average='macro')
    print f1_score(test_labels, preds, average='micro')
    print f1_score(test_labels, preds, average='weighted')
    
    
##################################
if __name__== "__main__":    
    testresult_fn = MODEL_DIR + "/test_result.csv"
    
    testset = prepare_test_data()    
    
    testing()
