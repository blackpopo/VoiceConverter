class Pix2PixConfig:
    def __init__(self):
        self.KAPPA = 0
        self.ADV = 0
        self.LAMBDA = 100
        #morethan datasize
        self.BUFFER_SIZE = 10000
        #なんか、BATCH size が　2以上にならん…。256にコードするとか？
        self.BATCH_SIZE = 64 #Use Instance Normalization
        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.EPOCHS = 200

class SpadeConfig:
    def __init__(self):
        self.BATCH_SIZE = 1
        self.EPOCHS = 400
        self.ADV_WEIGHT = 1
        self.FEATURE_WEIGHT = 10
        self.KL_WIGHT = 0.05
        self.CH = 64
        self.BETA1 = 0.5
        self.BETA2 = 0.999
        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.SEGMAP_CHANNELS = 1

class CycleGANConfig:
    def __init__(self):
        self.ADV = 0
        self.LAMBDA = 100
        #morethan datasize
        self.BUFFER_SIZE = 10000
        #なんか、BATCH size が　2以上にならん…。256にコードするとか？
        self.BATCH_SIZE = 8
        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.EPOCHS = 300

class SRConfig:
    def __init__(self):
        self.ADV = 0
        self.LAMBDA = 100
        #morethan datasize
        self.BUFFER_SIZE = 10000
        #なんか、BATCH size が　2以上にならん…。256にコードするとか？
        self.BATCH_SIZE = 1
        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.EPOCHS = 200

class SPConfig:
    def __init__(self):
        self.LAMBDA = 100
        #morethan datasize
        self.BUFFER_SIZE = 10000
        #なんか、BATCH size が　2以上にならん…。256にコードするとか？
        self.BATCH_SIZE = 8
        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.EPOCHS = 200
