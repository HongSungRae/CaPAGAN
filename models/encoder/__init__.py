from models.encoder import encoder

__all__ = ['encoder']

# library
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
import gc

# local
from datasets.glas2015 import GlaS2015
from utils import misc
from models.encoder.encoder import Encoder


class EncoderTrainer():
    def __init__(self, path, exp_name_encoder, exp_name_cp, data,
                 width,
                 original, cp, imsize,
                 batch_size, shuffle,
                 epochs, lr):
        '''
        = input =
        path : abs path of main.py
        exp_name_encoder : Name this encoder.
        exp_name_cp : If encoder is trained using CP samples, plz indicate cp sample experiment name.
        data : in [GlaS2015, Seegene, CRAG]
        width : width of ResNet in [34,50]
        original : if you want to use original "data" -> True
        cp : if you want to use "exp_name_cp" of cp samples -> True
        '''
        # given variables
        self.path = path
        self.exp_name_encoder = exp_name_encoder
        self.exp_name_cp = exp_name_cp
        self.original = original
        self.cp = cp
        self.batch_size= batch_size
        self.shuffle = shuffle
        self.imsize = imsize
        self.epochs = epochs
        self.width = width

        # other initial variable
        self.train_dataloader, self.test_dataloader = self.get_dataset(data)
        self.encoder = Encoder(self.num_classes, width)

        if torch.cuda.is_available():
            self.encoder = Encoder(self.num_classes, width).cuda()
            self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
            self.crietrion = nn.CrossEntropyLoss()
        else:
            raise RuntimeError("\n Alert : No available GPUs")

        # save path to save model pt file and meam/std.yaml
        misc.make_dir(fr'{self.path}/data/bank/encoder/{exp_name_encoder}')

        # log
        configuration = {'original':original, 'cp':cp, 'data':data, 'width':width, 'imsize':imsize,
                         'exp_name_cp':exp_name_cp, 'exp_name_encoder':exp_name_encoder}
        misc.save_yaml(fr'{self.path}/data/bank/encoder/{exp_name_encoder}/configuration.yaml', configuration)
        self.log = {}


    def get_dataset(self, data):
        if data == 'GlaS2015':
            train_dataset = GlaS2015(path=self.path,
                               original=self.original,
                               exp_name_cp=self.exp_name_cp,
                               imsize=self.imsize)
            test_dataset = GlaS2015(path=self.path,
                                    original=True,
                                    split='test',
                                    imsize=self.imsize)
            self.num_classes = 2
            self.task = 'binary'
        elif data == 'CRAG':
            pass
        elif data == 'Seegene':
            pass
        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle)
        test_dataloader = DataLoader(test_dataset, self.batch_size, False)
        return train_dataloader, test_dataloader
    

    def train(self):
        # log
        loss_list = []
        acc_list = []
        
        # train
        milestone = [int(self.epochs*inter*0.1) for inter in range(0,11) ]
        for epoch in tqdm(range(self.epochs), desc=' Alert : Training encoder...'):
            acc_epoch_list = []
            loss_epoch_list = []
            for i, (image, _, grade) in enumerate(self.train_dataloader):
                image, grade = image.cuda(), grade.cuda()
                grade_pred = self.encoder(image)
                loss = self.crietrion(grade_pred, grade)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accuracy = torchmetrics.functional.accuracy(torch.argmax(grade_pred,-1).detach().cpu(), 
                                                            grade.detach().cpu(), 
                                                            self.task, 
                                                            num_classes=self.num_classes, 
                                                            top_k=1)
                accuracy = round(accuracy.item(),3)
                acc_epoch_list.append(accuracy)
                loss_epoch_list.append(round(loss.detach().cpu().item(),3))
            loss_list.append(round(sum(loss_epoch_list)/len(loss_epoch_list),3))
            acc_list.append(round(sum(acc_epoch_list)/len(acc_epoch_list),3))
            if epoch in milestone:
                valid_accuracy = self.validation(self.encoder)
                train_accuracy = round(sum(acc_epoch_list)/len(acc_epoch_list),3)
                print(f'\n [Epoch / Epochs] | [{epoch} / {self.epochs}] : {loss_list[-1]} | {train_accuracy} | {valid_accuracy}')
        
        # memory clear
        del image, grade, grade_pred, loss
        self.optimizer.zero_grad()
        self.crietrion.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

        # save
        torch.save(self.encoder.state_dict(), fr'{self.path}/data/bank/encoder/{self.exp_name_encoder}/encoder.pt')
        self.log['train_loss'] = loss_list
        self.log['accuracy'] = acc_list
        misc.save_yaml(f'{self.path}/data/bank/encoder/{self.exp_name_encoder}/log.yaml', self.log)
        del loss_list, acc_list

    def validation(self, encoder):
        valid_accuracy_list = []
        with torch.no_grad():
            for _, (image, _, grade) in enumerate(self.test_dataloader):
                image, grade = image.cuda(), grade.cuda()
                grade_pred = encoder(image)
                accuracy = torchmetrics.functional.accuracy(torch.argmax(grade_pred,-1).detach().cpu(), 
                                                            grade.detach().cpu(), 
                                                            self.task, 
                                                            num_classes=self.num_classes, 
                                                            top_k=1)
                valid_accuracy_list.append(round(accuracy.item(),3))

        return round(sum(valid_accuracy_list)/len(valid_accuracy_list),3)

    def inference(self):
        # load model again
        del self.encoder
        torch.cuda.empty_cache()
        self.encoder = Encoder(self.num_classes, self.width).cuda()
        self.encoder.load_state_dict(torch.load(f'{self.path}/data/bank/encoder/{self.exp_name_encoder}/encoder.pt'))

        # accumulate
        accum = []
        for i in range(self.num_classes):
            accum.append([])
        with torch.no_grad():
            for i, (image, _, grade) in tqdm(enumerate(self.train_dataloader), desc='\n Alert : On inference...'):
                image = image.cuda()
                grade = grade.detach().cpu().tolist()
                feature_vector = self.encoder.net(image)
                for idx, g in enumerate(grade):
                    accum[g].append(feature_vector[idx].detach().cpu())

        # get mean and std
        mean_std = {}
        for i in range(self.num_classes):
            mean = torch.mean(torch.stack(accum[i]), dim=0).tolist()
            std = torch.std(torch.stack(accum[i]), dim=0).tolist()
            mean_std[i] = [mean, std]
        
        # save
        misc.save_yaml(f'{self.path}/data/bank/encoder/{self.exp_name_encoder}/mean_std.yaml', mean_std)

        # memory clear
        del self.encoder, image, grade, feature_vector, accum, mean_std
        torch.cuda.empty_cache()
        gc.collect()


    def __call__(self):
        self.train()
        self.inference()