

class config():
    def __init__(self):
        self.data_path = 'archive//data'
        self.train_path = self.data_path+'//train'
        self.test_path = self.data_path+'//test'
        self.val_path = self.data_path+'//val'
        self.lr = 1e-4
        self.batch_size = 16
        self.save_model = self.data_path + '//resnet18.ckpt'