import torch
import torch.nn as nn



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = torch.tensor(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.tensor(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):        
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.to(torch.device('cuda')))
    

class GeneratorLoss(nn.Module):
    def __init__(self, fm_loss=False, return_all=False, lam=0):
        super().__init__()
        self.return_all = return_all
        self.fm_loss = fm_loss
        self.lam = lam
        self.l1 = nn.L1Loss()

    def forward(self, pred_fake, layer_outputs_real=None, layer_outputs_fake=None):
        loss_fake = torch.mean((pred_fake-1)**2)
        if self.fm_loss:
            assert layer_outputs_real is not None, 'Please get layer_outputs from Discriminator.layer_outputs'
            assert layer_outputs_fake is not None, 'Please get layer_outputs from Discriminator.layer_outputs'
            loss_fm = 0
            for key in layer_outputs_fake.keys():
                loss_fm += self.l1(layer_outputs_real[key], layer_outputs_fake[key])
            if self.return_all:
                return loss_fake + self.lam*loss_fm, loss_fake, loss_fm
            else:
                return loss_fake + loss_fm
        else:
            return loss_fake


class DiscrimonatorLoss(nn.Module):
    def __init__(self, return_all=False):
        super().__init__()
        self.return_all = return_all

    def forward(self,pred_real, pred_fake):
        loss_real = torch.mean((pred_real-1)**2)
        loss_fake = torch.mean((pred_fake)**2)
        loss = loss_real + loss_fake
        if self.return_all:
            return loss, loss_real, loss_fake
        else:
            return loss


if __name__ == '__main__':
    pass
    # discriminator = SPADEDiscriminator(None)
    # criterion_dis = DiscrimonatorLoss()
    # criterion_gen = GeneratorLoss(fm_loss=False)

    # img_real = nn.functional.tanh(torch.randn((4,3,512,512)))
    # img_fake = nn.functional.tanh(torch.randn((4,3,512,512)))
    # mask_real = torch.ones((4,1,512,512))
    # mask_fake = torch.ones((4,1,512,512))

    # pred_real = discriminator(img_real, mask_real)
    # layer_outputs_real = discriminator.layer_outputs
    # discriminator.layer_outputs = {}

    # pred_fake = discriminator(img_fake, mask_fake)
    # layer_outputs_fake = discriminator.layer_outputs

    # loss_dis = criterion_dis(pred_real, pred_fake)
    # loss_gen = criterion_gen(pred_fake, layer_outputs_real, layer_outputs_fake)
    
    # print(f'D Loss : {loss_dis.item()}')
    # print(f'G Loss : {loss_gen.item()}')





    # criterion_dis = DiscrimonatorLoss()

    # pred_real = nn.functional.sigmoid(torch.randn((4,1)))
    # pred_fake = nn.functional.sigmoid(torch.randn((4,1)))

    # loss_dis = criterion_dis(pred_real, pred_fake)
    # print(loss_dis)