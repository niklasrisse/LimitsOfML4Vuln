import time
import numpy as np
from tqdm import tqdm

START_TOKEN  = -1
END_TOKEN  = -2

class MarkovModel:
    
    def __init__(self, corpus, words):
        self.corpus = corpus
        self.words = words
        self.unigram, self.bigram, _ = self.calculate_grams()
        self.transition_probs = self.calculate_transition_probs()
        self.priors = self.unigram / np.sum(self.unigram)

    def calculate_grams(self):

        sentences = self.corpus
    
        start = time.time()
        
        unigram = np.zeros(shape = self.words.shape[0], dtype = np.int32)
        bigram = np.zeros(shape = (self.words.shape[0], self.words.shape[0]), dtype = np.int32)

        progress_bar = tqdm(range(len(sentences)))
        
        for sentence in sentences:
            n_words = len(sentence)

            bigram[0, np.where(self.words == sentence[0])[0]] += 1
            
            for i in range(n_words):
                unigram[np.where(self.words == sentence[i])[0]] += 1
                if (i < n_words - 1):
                    bigram[np.where(self.words == sentence[i])[0], np.where(self.words == sentence[i+1])[0]] += 1
                elif (i == n_words - 1):
                    bigram[np.where(self.words == sentence[i])[0], len(self.words)-1] += 1
            
            unigram[0] += 1
            unigram[-1] += 1
            
            progress_bar.update(1)
        
        end = time.time()
        
        return unigram, bigram, end-start

    def calculate_transition_probs(self):

        transition_probs = self.bigram.astype(np.float64)
        
        for i, row in enumerate(transition_probs):
            if (self.unigram[i] != 0):
                transition_probs[i] = row / self.unigram[i]

        return transition_probs

    def inject_noise(self, size):

        self.transition_probs += size

        for i, row in enumerate(self.transition_probs):
            self.transition_probs[i] = self.transition_probs[i] / np.sum(row)
        print(size)

    def sample(self, maxlength = 10):
        
        sample = [START_TOKEN]
        
        for i in range(maxlength):
            candidate = np.random.choice(self.words, p = self.transition_probs[np.where(self.words == sample[-1])[0]][0])
            if (candidate == END_TOKEN):
                break
            elif (candidate == START_TOKEN):
                continue
            sample.append(candidate)
            
        sample.remove(START_TOKEN)
            
        probability = self.calculate_probability_of_sample(sample)

        return sample, probability
    
    def calculate_cross_entropy_of_sample(self, sentence):
        
        words = sentence[:]
        words.append(END_TOKEN)
        words.insert(0, START_TOKEN)
        ce = 0
        
        for i, word in enumerate(words): 
            if i > 0:
                if not np.where(self.words == word)[0]:
                    ce = 0 
                    break
                else:
                    ce += np.log(self.transition_probs[np.where(self.words == words[i-1])[0], np.where(self.words == word)[0]][0])
            
        return -1/len(words) * ce
    
    
        
        