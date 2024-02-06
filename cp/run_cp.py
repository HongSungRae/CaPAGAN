# library
from tqdm import tqdm
import pandas as pd
import random
import cv2
import numpy as np
import os

# local
from utils import misc
from cp.cp import cp


def run_cp(path:str=None, exp_name:(str, int)=None, n_samples:int=100, dataset:str='', imsize:int=512, self_mix=False):
    '''
    = input =
    path : abs path of main.py
    exp_name : please name the experiment
    n_samples : the number of samples that user want to synthesize by CP
    dataset : dataset
    imsize : image size
    self_mix : if True image_source == image_target. i.e. self-oriented-mix
    
    = output =
    It saves (image_cp, mask_cp, annotation_df) locally.
    Save path is "{path}/data/bank/cp/{exp_name}".

    = return =
    Nothing to return
    '''
    assert path is not None
    assert exp_name is not None
    assert n_samples >= 1
    assert dataset in ['GlaS2015', 'CRAG', 'Seegene']
    assert (imsize >= 256) and (imsize%(2**4) == 0)

    # make save path
    misc.make_dir(fr'{path}/data/bank/cp/{exp_name}')

    # save configuration
    configuration = {'exp_name_cp':exp_name, 'n_samples':n_samples, 'dataset':dataset, 'imsize':imsize, 'self_mix':self_mix}
    misc.save_yaml(fr'{path}/data/bank/cp/{exp_name}/configuration.yaml', configuration)
    
    # load data location
    sample_dic = get_sample_dic(path, dataset) # {class_k : [[img_j_name, mask_j_name], ..., [img_l_name, mask_l_name]]}

    # to make annotation information DataFrame
    name_list = []
    grade_list = []

    # synthesize new samples
    for i in tqdm(range(n_samples), desc='generating CP samples...'):
        ## select source, target data
        grade = random.choice(list(sample_dic.keys()))
        grade_list.append(grade)
        name_list.append(f'{i}.png')
        img_mask_list = sample_dic[grade]
        source_idx, target_idx = random.sample([k for k in range(len(img_mask_list))], k=2)
        if self_mix:
            image_source_name, mask_source_name = img_mask_list[source_idx]
            image_target_name, mask_target_name = image_source_name, mask_source_name
        else:
            image_source_name, mask_source_name = img_mask_list[source_idx]
            image_target_name, mask_target_name = img_mask_list[target_idx]
        
        ## load data
        image_source = cv2.imread(f'{path}/data/{image_source_name}', cv2.IMREAD_COLOR) # (h, w, 3)
        mask_source = cv2.imread(f'{path}/data/{mask_source_name}', cv2.IMREAD_GRAYSCALE) # (h, w)
        image_target = cv2.imread(f'{path}/data/{image_target_name}', cv2.IMREAD_COLOR)
        mask_target = cv2.imread(f'{path}/data/{mask_target_name}', cv2.IMREAD_GRAYSCALE)

        ## data preprocessing
        mask_source = np.where(mask_source>=1.0, 1.0, 0.0).astype(np.uint8)
        mask_target = np.where(mask_target>=1.0, 1.0, 0.0).astype(np.uint8)
        
        ## Do CP
        try:
            image_cp, mask_cp = cp(image_source, image_target, mask_source, mask_target, imsize)

            ## save image and mask
            cv2.imwrite(f'{path}/data/bank/cp/{exp_name}/{i}.png', image_cp)
            cv2.imwrite(f'{path}/data/bank/cp/{exp_name}/{i}_mask.png', mask_cp)
        except:
            continue

    ## save annotation information
    annotation_df = pd.DataFrame(data={'name':name_list, 'grade':grade_list})
    annotation_df.to_csv(f'{path}/data/bank/cp/{exp_name}/annotation.csv', index=False)
    
    print('\n Alert : CP augmentation finished...')



def get_sample_dic(path:str, dataset:str):
    '''
    = input =
    path : abs path of main.py
    dataset : dataset

    = return =
    sample_dic : {class_1 : [[img_1_name, mask_1_name], ..., [img_i_name, mask_i_name]],
                  ...
                  class_k : [[img_j_name, mask_j_name], ..., [img_l_name, mask_l_name]]}
    '''
    assert dataset in ['GlaS2015', 'Seegene', 'CRAG']
    
    sample_dic = {}

    if dataset == 'GlaS2015':
        train_list = [file for file in os.listdir(fr'{path}/data/Warwick QU Dataset (Released 2016_07_08)') if 'train' in file]
        grade_df = pd.read_csv(f'{path}/data/Warwick QU Dataset (Released 2016_07_08)/Grade.csv')
        class_0, class_1 = [], []
        for image in train_list:
            if 'anno' in image: # if mask
                continue
            grade = {' benign':0, ' malignant':1}[grade_df[grade_df['name']==image.rstrip('.bmp')].iloc[0,2]]
            mask = f'Warwick QU Dataset (Released 2016_07_08)/{image.rstrip(".bmp")+"_anno.bmp"}'
            image = f'Warwick QU Dataset (Released 2016_07_08)/{image}'
            image_mask = [image, mask]
            class_0.append(image_mask) if grade==0 else class_1.append(image_mask)
        sample_dic[0] = class_0
        sample_dic[1] = class_1
    elif dataset == 'CRAG':
        pass
    elif dataset == 'Seegene':
        pass

    return sample_dic



if __name__ == '__main__':
    pass