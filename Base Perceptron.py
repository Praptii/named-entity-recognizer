# Setup
# Part 1
import collections

def data_creator(filename):

    with open(filename,encoding="utf-8") as f:
        lines = f.readlines()
    
    lines = [x.replace('\t', ' ').replace('\n','') for x in lines]
    sentences = []

    sentence = ''
    tag_line = []

    for line in lines:

        if line == '':
            sentences.append([sentence, tag_line])
            sentence = ''
            tag_line = []
            continue

        splits = line.split(' ')
        sentence += splits[0]  + ' '
        tag_line.append(splits[1])
        
    return sentences

train_data = data_creator('train')
dev_data = data_creator('dev')
test_data = data_creator('test')

total_tags = []
from collections import Counter

for sentence in train_data:
    total_tags.extend(sentence[-1])
    
tag_types = list(Counter(total_tags).keys())   

# Part 3
# Perceptron Model


class Perceptron(object):
    """Base Perceptron model class"""

    def __init__(self, tag_types):
        self.weights = collections.Counter()
        self.tag_types = tag_types

    def get_features(self, prevTag, tag, word=None):
        '''Given a previous tag, current tag, and current word, generate a list of features.  Note that word may be omitted for <EOS>'''
        if word is not None:
            features_list = [(prevTag, tag), (tag, word)]
        else:
            features_list = [(prevTag, tag)]
        return features_list

    def score_features(self, features_list):
        '''Given a list of features, compute the score from those features'''
        score = 0
        for feat in features_list:
            score += self.weights[feat]
        return score

    def train(self, sents, epochs=7):
        '''Given a list of sentences in the form:
        ['space separated string', ['O', 'O', 'O']]
        Train the perceptron for the given number of epochs'''
        for epoch in range(epochs):
#             random.shuffle(sents) # Modification
            for sent, tags in sents:            
                self.train_line(sent, tags)

    def train_line(self, sent, tags):
        '''Trains from a single sentence.  
        sent is a space separated string
        tags is a list of correct tags'''
        mytags = self.viterbi(sent)
        prevCorrect = '<BOS>'
        prevPred = '<BOS>'
        for w, c, p in zip(sent.split(' '), tags, mytags):
            if c != p:
                
                for feat in self.get_features(prevCorrect, c, w):
                    self.weights[feat] += 1
                for feat in self.get_features(prevPred, p, w):
                    self.weights[feat] -= 2
                    

            prevCorrect = c
            prevPred = p
        
            
        if c != p:
            # If the final tag is wrong, also update <EOS>
            for feat in self.get_features(c, '<EOS>'):
                self.weights[feat] += 1
            for feat in self.get_features(p, '<EOS>'):
                self.weights[feat] -= 2

    def tag_sents(self, sents, outFile='dev-percep.out'):
        '''Given a list of sentences in the form ['sample sentence here', ['O', 'O', 'O'] ], predicts tag sequence and writes to file for scoring. '''
        with open(outFile,'w', encoding='utf-8') as g:
            for sent, tags in sents:
                mytags = self.viterbi(sent)
                for s, c, m in zip(sent.split(' '), tags, mytags):
                    g.write(s + ' ' + c + ' '+ m + '\n')
                g.write('\n')
                
    def generate_viterbi_map(self, emission_keys, transition_keys):
        viterbi_map = {}
    
        
        for t_key in transition_keys:
            for e_key in emission_keys:
                viterbi_map[(t_key, e_key)] = -10000
        
        return viterbi_map
        

    def viterbi(self, sent):
        '''Given a space separated string as input, produce the 
        highest weighted tag sequence for that string.'''
        # You will replace this, for now it just returns 'O' for every tag
        sent_split = sent.split(' ')
        
        emission_keys = [(x,y) for x in sent_split for y in sent_split]
        transition_keys = [(x,y) for x in self.tag_types for y in self.tag_types]
        

        emission_map = {key: 0 for key in emission_keys}
        transition_map = {key: 0 for key in transition_keys}
        
        viterbi_map = self.generate_viterbi_map(sent_split, self.tag_types)
        
        path = []
        
        prev_word = '' 
        prev_tag = '<BOS>'
        
        prev_max_score = -10000
        prev_max_viterbi_elem = '' 
        
        for word in sent_split:
            iter_pre_tag = ''
            iter_max_score = -10000
            
            for tag in self.tag_types:
                viterbi_map[(tag, word)] = self.weights[(tag, word)] + self.weights[(prev_tag, tag)]
                score = self.score_features(self.get_features(prev_tag, tag, word))
    
                if score > iter_max_score:
                    iter_max_score = score
                    iter_pre_tag = tag
            
            prev_tag = iter_pre_tag
            path.append(prev_tag)
            prev_max_score += iter_max_score
            
        return path

perceptron = Perceptron(tag_types)
train_data = data_creator('train')

perceptron.train(train_data, 6)
perceptron.tag_sents(test_data, f'dev-percep.out')