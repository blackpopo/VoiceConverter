import random
import numpy as np

class Dataset:
    def __init__(self, source_data, target_data):
        self.source_data = np.array(source_data)
        self.target_data = np.array(target_data)
        assert self.source_data.shape[0] == self.target_data.shape[0]
        self.length = self.source_data.shape[0]

    def shuffle(self):
        index = list(range(self.source_data.shape[0]))
        random.shuffle(index)
        temp_src = [self.source_data[i] for i in index]
        temp_tgt = [self.target_data[i] for i in index]
        self.source_data = np.array(temp_src)
        self.target_data = np.array(temp_tgt)

    def batch(self, batch_size):
        iter =   self.length // batch_size
        temp_source, temp_target = list(), list()
        for i in range(iter):
            temp_source.append(self.source_data[i * batch_size: (i+1)*batch_size])
            temp_target.append(self.target_data[i * batch_size: (i+1) * batch_size])
        if self.length % batch_size != 0:
            temp_source.append(self.source_data[self.length//batch_size * batch_size:])
            temp_target.append(self.target_data[self.length // batch_size * batch_size:])
        self.source_data = np.array(temp_source)
        self.target_data = np.array(temp_target)

    def get(self):
        return [self.source_data, self.target_data]

if __name__=='__main__':
    a = list(range(0, 20))
    ds = Dataset(a, a)
    ds.shuffle()
    print(ds.get())
    ds.batch(7)
    print(ds.get())


