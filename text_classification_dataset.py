import random
from random import randint, randrange
from glob import glob
from math import floor, ceil

class TextClassificationDatasetCreator(object):
    def __init__(self,seed=42):
        self.dataset = dict(train=[],
                            val=[],
                            test=[])
        self.train = {'text':[],'labels':[]}
        self.val = {'text':[],'labels':[]}
        self.test = {'text':[],'labels':[]}
        self.seed = seed
        random.seed(seed)
        self.patient_utterances = []
        self.labels = []
        self.label_names = ["[{'function':None}]","other", "1", "2", "letters","same","unsure","repeat"]
    '''
    FILE OPERATIONS.
    '''
    def read_captions(self, path):
        self.fname = path
        captions =  open(path).read().splitlines()
        captions = [i.strip() for i in captions]
        if captions[-1] == '':
            captions = captions[:-2]
        return captions
    '''
    LABEL ASSIGNMENT
    '''
    def label_patient_utterances(self,txt):
        num_lines = len(txt)
        for idx in range(num_lines):
            if txt[idx].startswith("[Patient__]"):
                if txt[idx+1].startswith("[Phoropter]"):
                    label = self.find_label(txt[idx+1])
                    if label:
                        if txt[idx][13:] not in self.patient_utterances:
                            self.patient_utterances.append(txt[idx][13:])
                            self.labels.append(label)

    def find_label(self,txt):
        new_labels = self.label_names
        if txt[13:] == new_labels[0]:
            return new_labels[1] 
        if '1' in txt[13:]:
            return new_labels[2]
        if '2' in txt[13:]:
            return new_labels[3]
        if 'accuracy' in txt[13:]:
            return new_labels[4]
        if 'same' in txt[13:]:
            return new_labels[5]
        else:
            print(f"no match for line: {txt}")

    def label2id(self):
        candidate_labels = ["1", "2", "same", "letters", "other"]
        self.num_classes = len(candidate_labels)
        ints = range(len(candidate_labels))
        mapping = dict(zip(candidate_labels,ints))
        mapping = {k:v for k,v in sorted(mapping.items(), key= lambda x:x[0])}
        try:
            for idx in range(len(self.labels)):
                self.labels[idx] = mapping[self.labels[idx]]
                self.labels[idx] = self.onehot(self.labels[idx])
        except KeyError:
            print(f'ERROR\n{mapping.items()}')
            print(self.labels[idx])

    def onehot(self,label):
        new = [0 for i in range(self.num_classes)]
        new[label-1] = 1
        return new

    def dset(self, split, data_dir):
        self.patient_utterances.clear()
        self.labels.clear()
        paths = sorted(glob(data_dir + '/*.txt'))
        print(f'{len(paths)} files found.')
        try:
            for path in paths:
                captions = self.read_captions(path)
                self.label_patient_utterances(captions)
            print(f'num utterances: {len(self.patient_utterances)} labels: {len(self.labels)}')
            self.label2id()
        except BaseException as e:
            print(f'Error with file: {path}')
            print(e)
        if split == 'train':
            self.train = {'text':self.patient_utterances.copy(),
                          'label':self.labels.copy()}
            return self.train
        if split == 'val':
            self.val = {'text':self.patient_utterances.copy(),
                        'label':self.labels.copy()}
            return self.val
        if split == 'test':
            self.test = {'text':self.patient_utterances.copy(),
                         'label':self.labels.copy()}
            return self.test