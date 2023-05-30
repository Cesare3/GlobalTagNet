from os import PathLike
from typing import List


class Vocab:
    def __init__(self, filename: PathLike, idf_filename: str = None, n_tokens: int = None):
        super().__init__()
        self.filename = filename
        with open(filename) as fh:
            words = [word.strip() for word in fh]
        if idf_filename is not None:
            with open(idf_filename) as fh:
                idfs = [float(line.strip()) for line in fh]
        else:
            idfs = None
        if n_tokens is not None:
            words = words[:n_tokens - 2]  # -2 for OOV_token, PAD_token
            if idfs is not None:
                idfs = idfs[:n_tokens - 2]  # -2 for OOV_token, PAD_token
                idfs = [1.0, 1.0] + idfs  # for OOV_token, PAD_token
        self.idfs = idfs
        start = int(self.oov_token not in words) + int(self.pad_token not in words)
        self._word2id = {
            word: index
            for index, word in enumerate(words, start=start)  # left some indices for OOV_token, PAD_token
        }
        self._word2id[self.pad_token] = self.pad_index
        self._word2id[self.oov_token] = self.oov_index
        self._id2word = {
            v: k for k, v in self._word2id.items()
        }
    
    @property
    def oov_token(self):
        return "<OOV>"
    
    @property
    def oov_index(self):
        return 1
    
    @property
    def pad_token(self):
        return "<PAD>"
    
    @property
    def pad_index(self):
        return 0
            
    def __len__(self):
        return len(self._word2id)
    
    def word2id(self, word: str):
        if word in self._word2id:
            return self._word2id[word]
        else:
            return self.oov_index
    
    def id2word(self, index: str):
        if index in self._id2word:
            return self._id2word[index]
        else:
            return self.oov_token
        
    def convert_text_to_word_bag(self, text: str) -> List[int]:
        word_bags = [0] * len(self)
        for word in text.lower().strip().split():
            word_bags[self.word2id(word)] = 1
        return word_bags
    
    def convert_texts_to_word_bags(self, texts: List[str]) -> List[List[int]]:
        word_bags = [self.convert_text_to_word_bag(text) for text in texts]
        return word_bags
    
    def convert_text_to_word_indices(self, text: str, n_seq: int) -> List[int]:
        word_indices = [self.pad_index] * n_seq
        for index, word in enumerate(text.lower().strip().split()):
            if index >= n_seq:
                break
            word_indices[index] = self.word2id(word)
        return word_indices
    
    def convert_texts_to_word_indices(self, texts: List[str], n_seq: int) -> List[List[int]]:
        return [
            self.convert_text_to_word_indices(text, n_seq) for text in texts
        ]
    
    def convert_text_to_tfidf(self, text: str) -> List[float]:
        tf = [0] * len(self)
        for word in text.lower().strip().split():
            tf[self.word2id(word)] += 1
        tfidf = [
            tf_ * idf_ for tf_, idf_ in zip(tf, self.idfs)
        ]
        return tfidf
    
    def convert_texts_to_tfidfs(self, texts: List[str]) -> List[List[float]]:
        tfidfs = [self.convert_text_to_tfidf(text) for text in texts]
        return tfidfs


if __name__ == "__main__":
    vocab = Vocab("/Users/saner/Desktop/第二节课/project/AAPD/vocab.txt")
    print(len(vocab))
    print(vocab.oov_token, vocab.oov_index)
    print(vocab.word2id("the"), vocab.id2word(vocab.word2id("the")))
    print(vocab.word2id("deepeye"), vocab.id2word(vocab.word2id("deepeye")))
