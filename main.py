# library
import os
import warnings
import torch



# local
from utils import misc
from utils.args import get_parser
from cp import run_cp
from models.encoder import EncoderTrainer
from models.spade import SPADETrainer


# variables
path = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)



def main(args):
    # CP
    if (args.exp_name_cp is not None) and (not os.path.exists(fr'{path}/data/bank/cp/{args.exp_name_cp}')):
        run_cp.run_cp(path=path,
                      exp_name=args.exp_name_cp,
                      n_samples=args.n_samples_cp,
                      dataset=args.dataset_cp,
                      imsize=args.imsize_cp,
                      self_mix=args.self_mix)

    # Encoder
    if (args.exp_name_encoder is not None) and (not os.path.exists(fr'{path}/data/bank/encoder/{args.exp_name_encoder}')):
        ## load trainer
        encoderTrainer = EncoderTrainer(path=path,
                                        exp_name_encoder=args.exp_name_encoder,
                                        exp_name_cp=args.exp_name_cp,
                                        data=args.dataset_encoder,
                                        width=args.width,
                                        original=args.original,
                                        cp=args.cp,
                                        imsize=args.imsize_encoder,
                                        batch_size=args.batch_size_encoder,
                                        shuffle=True,
                                        epochs=args.epochs_encoder,
                                        lr=args.lr_encoder)
        ## train and save mean_std
        encoderTrainer()
        
    # GAN
    if (args.exp_name_gan is not None) and (not os.path.exists(fr'{path}/data/bank/gan/{args.exp_name_gan}')):
        ## init
        misc.make_dir(f'{path}/data/bank/gan/{args.exp_name_gan}')

        ## load trainer
        spadeTrainer = SPADETrainer(path, args)

        ## train and inference
        spadeTrainer.__call__()



if __name__ == '__main__':
    args = get_parser()
    main(args)