import random
from random import randint, randrange
from glob import glob
from math import floor, ceil

class VisionDatasetCreator(object):
    '''
    Class for turning captioned dialogues into a dataset compatible
    with Huggingface training tools.
    Randomly samples lines of text from clean caption files.
    From each file is first divided into 4 sections:
    gs - greeting & setup
    sp - spherical refinement
    ac - axis & cylinder refinement
    cv - close vision & valediction
    The average percentage that each exam phase, based on 30 dialogue samples
    is used to drive the sampling of each text document so that dataset samples
    are more consistent with the average real exam.
    '''
    def __init__(self, sampling_strategy:dict, assistant=True, seed=randint(0,10000)):
        self.dataset = dict(train=[],
                            val=[],
                            test=[])
        self.seed = seed
        random.seed(seed)
        self.sampling_strategy = sampling_strategy
        self.assistant = assistant

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
    SAMPLING
    '''
    def check_sample_length(self, cap_len, start, end, max_len):
        # assure sample is within the bounds of a caption list.
        cnt = 0
        while end > (cap_len-1):
            cnt +=1
            # select new start
            start -= randint(2,max_len)
            end = start + randint(2, max_len)
            if cnt > 5:
                print('while loop in check_sample_length got stuck.')
                return
        return start, end
    
    def find_assistant_idxs(self,captions:list):
        '''
        finds indices beginning with [Assistant] in captions.
        '''
        idxs = []
        for idx, txt in enumerate(captions):
            if txt.startswith("[Assistant]"):
                idxs.append(idx)
        return idxs
    
    def find_phoropter_idxs(self,captions:list):
        '''
        finds indices beginning with [Phoropter] in captions.
        '''
        idxs = []
        for idx, txt in enumerate(captions):
            if txt.startswith("[Phoropter]"):
                idxs.append(idx)
        return idxs

    def divide_data(self,captions:list):
        '''
        sampling strategy is unpacked purposely for clarity.
        '''
        # avg percentages of exam phase lengths
        gs=self.sampling_strategy['gs'][0]
        sp=self.sampling_strategy['sp'][0]
        ac=self.sampling_strategy['ac'][0]
        cv=self.sampling_strategy['cv'][0]
        # minumum dialogue lengths
        gs_min = self.sampling_strategy['gs'][2]
        sp_min = self.sampling_strategy['sp'][2]
        ac_min = self.sampling_strategy['ac'][2]
        cv_min = self.sampling_strategy['cv'][2]
        
        s_len = len(captions)
        gs_range = floor(gs*s_len)
        sp_range = ceil(sp*s_len)
        ac_range = ceil(ac*s_len)
        cv_range = max(cv_min, floor(cv*s_len))
        sp_end = gs_range + sp_range
        ac_end = sp_end + ac_range
        gs_sample = captions[:gs_range]
        sp_sample = captions[gs_range:sp_end]
        ac_sample = captions[sp_end:ac_end]
        cv_sample = captions[ac_end:]
        return gs_sample, sp_sample, ac_sample, cv_sample
    
    def cycler(self, val, interval):
        return val % interval
    
    def select_sample(self,phase):
        '''
        The sampling strategy begins by randomly selecting a start.
        The end point is a randomly selected point satisfying 
        min_len < end < max_len
        The start and end points are checked so they don't exceed
        the bounds of the caption data.
        If the sample is within bounds, we assure that the Assistant
        response is the end of the sample.
        '''
        if self.assistant == True:
            idxs = self.find_assistant_idxs(self.sampling_strategy[phase][4])
        else:
            idxs = self.find_phoropter_idxs(self.sampling_strategy[phase][4])
        min_dist = self.sampling_strategy[phase][2]
        max_dist = self.sampling_strategy[phase][3]
        # randomly select assistant response index
        z = randrange(len(idxs))
        end = idxs[z]
        # get a starting point
        start = end - randint(min_dist, max_dist)
        cnt = 0
        while len(self.sampling_strategy[phase][4][start:end+1]) < min_dist:
            start = end - randint(min_dist, max_dist)
            cnt += 1
            if cnt > 3:
                # print('Trying new end')
                z = randrange(len(idxs))
                end = idxs[z]
            if cnt >5:
                # sometimes the loop hangs...
                break
        sample = self.sampling_strategy[phase][4][start:end+1]
        if len(sample) > max_dist:
            start = start + (len(sample) - max_dist)
            sample = self.sampling_strategy[phase][4][start:end+1]
            if len(sample) > max_dist:
                print(f'uh oh, {len(sample)}')
        if len(sample) > min_dist:
            return sample
    '''
    TRAINING DATA FORMATTING
    '''
    def to_dialogue(self, txt):
        '''
        reformats the captioned text into LLM input format.
        returns a dictionary with two keys containing a list
        of dictionaries in the following format:
        {"role":speaker, "content":content}
        '''
        output = dict(dialogue=[],
                      response=[])
        for l in txt[:-1]:
            speaker = l[1:10]
            content = l[13:]
            line = {"role":speaker, "content":content}
            output['dialogue'].append(line)
        response = txt[-1]
        speaker = response[1:10]
        content = response[13:]
        line = {"role":speaker, "content":content}
        output['response'].append(line)
        return output
    '''
    DATASET CREATION
    '''
    def load(self, data_dir:str, split:str, size:int):
        '''
        Given a directory with data files split into the following structure:

        root/
        ----train/
        ----val/
        ----test/

        We count the number of files in the split directory.
        To reach the desired size, we divide the size evenly by
        the number of files in the split directory.
        Then, for each file, we sample until the same number of times
        until the desired dataset size is reached.
        '''
        paths = glob(data_dir + split + '/*.txt')
        paths.sort()
        # files available
        fcnt = len(paths)
        # samples per file
        scnt = size // fcnt
        exam_phases = list(self.sampling_strategy.keys())
        print(f'{fcnt} files found. Sampling {scnt} times per file.')
        # for each file
        for idx in range(fcnt):
            print(f'sampling file: {paths[idx]}')
            # read data
            captions = self.read_captions(paths[idx])
            # divide data into the phases in sampling strategy
            gs_sample, sp_sample, ac_sample, cv_sample = self.divide_data(captions)
            total_len = 0
            # data divided into phases
            samples = [gs_sample, sp_sample, ac_sample, cv_sample]
            # check to see if redundancy occured
            # a minimal amount is expected at the moment
            for s in samples:
                total_len+=len(s)
            if total_len - len(captions) !=0:
                print('WARNING: sample length not equal to captions length!')
                print(f'samples length: {total_len}\tcaptions length: {len(captions)} ')
            # clear old samples
            for k in exam_phases:
                # if an older list of samples is present in the strategy, remove it.
                if len(self.sampling_strategy[k]) > 4:
                    self.sampling_strategy[k] = self.sampling_strategy[k][:4]
            # add data to sampling strategy
            for s, k in zip(samples,exam_phases):
                self.sampling_strategy[k].append(s)
            
            # running track of total dataset size
            sample_cnt = (idx+1)*scnt
            
            # counter in case loop hangs
            cnt = 0

            # sample until desired dataset size is reached
            while len(self.dataset[split]) < sample_cnt:
                # tell sampler which part of the exam to sample from
                i = self.cycler(cnt, len(exam_phases))
                # give captions, a miniumum caption length, and a maximum caption length
                sample = self.select_sample(exam_phases[i])
                # sometimes a loop gets stuck and doesn't return a value
                if sample:
                    datum = self.to_dialogue(sample)
                    self.dataset[split].append(datum)
                cnt += 1
                if cnt > sample_cnt*2:
                    print(f'dataset length while loop stuck: {len(self.dataset[split])} sample count: {sample_cnt}')
                    break
        print(f'Expected length of data: {size}')
        print(f'Actual length: {len(self.dataset[split])}')

class VisionDataset(object):
    '''
    Huggingface compatible data class that makes the data iterable and returns it's length.
    '''
    def __init__(self, data):
        self.data = [i for i in data]
    def __len__(self):
        return len(self.data)
    def get_max_len(self, format_func):
        max_len = 0
        for sample in self.data:
            inpt = format_func(sample)
            if len(inpt) > max_len:
                max_len = len(inpt)
        return max_len