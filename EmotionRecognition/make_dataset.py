import numpy as np

def get_files1():
    return None

def get_files2():
    return None

def _batch_parser_normalize95(input, target, who):
    input = (input - np.min(input)) / (np.max(input) - np.min(input))
    input = input * (10.0 ** 10 - 1) + 1 #scale 1 < 10**10
    target = target * (10.0 ** 10 -1) + 1
    input = np.log(input / 10.0 ** 5) / np.math.log(10.0) / 5.0 #scale 10**-5 < 10**5 >> -1 < 1
    target = np.log(target / 10.0 ** 5 ) / np.log(10.0) / 5.0
    return input, target, who

"""
Pathを全部取ってくる
wav のtriming
wavからlog.melに変換
tensorで保存

"""