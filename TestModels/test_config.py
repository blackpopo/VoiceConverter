class Pix2PixConfig:
    def __init__(self):
        self.LAMBDA = 100
        #morethan datasize
        self.BUFFER_SIZE = 100000
        #なんか、BATCH size が　2以上にならん…。256にコードするとか？
        self.BATCH_SIZE = 1
        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.EPOCHS = 150
        
class LSTMConfig:
    def __init__(self):
        self.BUFFER_SIZE = 65535
        self.EPOCHS = 10
        self.TIMESTEP = 1
        self.DEC_UNITS = 256
        self.OUTPUT_SIZE = 128
        self.ENC_UNITS = 256
        self.BATCH_SIZE = 4
        self.MAX_LENGTH = 128