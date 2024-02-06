import argparse



def get_parser():
    # init parser
    parser = argparse.ArgumentParser(description='CaPAGAN')

    # add parser variables
    ## cp
    parser.add_argument('--exp_name_cp', '--enc', default=None, type=str,
                        help='Please name cp training experiment.')
    parser.add_argument('--n_samples_cp', '--nsc' , default=None, type=int,
                        help='The number samples user wants to synthesize by CP.')
    parser.add_argument('--dataset_cp', default=None, type=str,
                        choices=['GlaS2015', 'CRAG', 'Seegene'],
                        help='Datasets that user wants to create.')
    parser.add_argument('--imsize_cp', default=512, type=int,
                        help='Image size of CP sythesized samples.')
    parser.add_argument('--self_mix', action='store_true',
                        help='Self mix CP. Default is False. To make True --self_mix')

    ## encoder
    parser.add_argument('--exp_name_encoder', '--ene', default=None, type=str,
                        help='Please name encoder training experiment.')
    parser.add_argument('--dataset_encoder', default=None, type=str,
                        choices=['GlaS2015', 'CRAG', 'Seegene'],
                        help='Datasets that user wants to embed.')
    parser.add_argument('--width', default=34, type=int,
                        choices=[34,50],
                        help='ResNet width')
    parser.add_argument('--original', action='store_true',
                        help='Will you use original dataset to train encoder? Default is False. To make True --original')
    parser.add_argument('--cp', action='store_true',
                        help='Will you use cp dataset to train encoder? Default is False. To make True --cp')
    parser.add_argument('--imsize_encoder', default=512, type=int,
                        help='Image size which is used to train encoder.')
    parser.add_argument('--batch_size_encoder', '--bse', default=32, type=int,
                        help='Batch size to train encoder.')
    parser.add_argument('--epochs_encoder', '--ee', default=50, type=int,
                        help='Epochs to trian encoder.')
    parser.add_argument('--lr_encoder', default=1e-3, type=float,
                        help='Learning rate to train encoder.')

    ## gan
    parser.add_argument('--exp_name_gan', '--eng', default=None, type=str,
                        help='Please name gan training experiment.')
    parser.add_argument('--dataset_gan', default=None, type=str,
                        choices=['GlaS2015', 'CRAG', 'Seegene'])
    parser.add_argument('--imsize_gan', default=512, type=str,
                        help='GAN image size')
    parser.add_argument('--n_samples_gan', '--nsg', default=100, type=int,
                        help='The number samples user wants to synthesize by GAN.')

    
    ## SPADE Generator
    parser.add_argument('--spade_filter', default=128, type=int,
                        help='Number of CNN filter(depth) in SPADE')
    parser.add_argument('--gen_input_size', default=256, type=int,
                        help='z vector dimension. If 256, shape of z gonna be (1,256).')
    parser.add_argument('--gen_hidden_size', default=16384, type=int,
                        help='16384=128x128. z vector is embedded to this size.'),
    parser.add_argument('--spade_kernel', default=3, type=int)
    parser.add_argument('--spade_resblk_kernel', default=3, type=int)

    ## train arguments
    parser.add_argument('--epochs_gan', type=int, default=100,
                        help='Epochs to train gan.')
    parser.add_argument('--batch_size_gan', '--bsg', type=int, default=4,
                        help='Epochs to train gan.')
    parser.add_argument('--lr_gen', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_dis', default=0.0004, type=float,
                        help='Learning rate of discriminator.')
    parser.add_argument('--distributed', '--d', action='store_true',
                        help='Multi GPU distributed training. Default is False. To make True --d')
    
    # gpu check
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id')
    parser.add_argument('--world_size', '--ws', default=2, type=int,
                        help='world size')
    parser.add_argument("--local_rank", default=0, type=int)

    # check parser variables
    args = parser.parse_args()
    check_parser(args)
    
    # return
    return args


def check_parser(args):
    ## cp
    
    ## SPADE Generator
    # assert args.gen_hidden_size%16 != 0, "Gen hidden size not multiple of 16"
    pass