from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.crops import extract_crops_z_numpy, extract_crops_x_numpy, pad_frame_numpy, resize_images
import cv2


videos_path = '/home/iair-imcs/Documents/paper/siamfc-tf-master/data/validation/training(1)'
start_frame = 0


def main():

    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    print ("final_score_sz is:%d" %(final_score_sz))
    gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_path) 
    num_frames = np.size(frame_name_list)
    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num) 
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    pos_x, pos_y, target_w, target_h = region_to_bbox(gt[start_frame])  


    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz
    scaled_exemplar = z_sz * scale_factors
    scaled_search_area = x_sz * scale_factors
    scaled_target_w = target_w * scale_factors
    scaled_target_h = target_h * scale_factors


    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz

    #search size
    x_sz0_ph = scaled_search_area[0]
    x_sz1_ph = scaled_search_area[1]
    x_sz2_ph = scaled_search_area[2]
    image = Image.open(frame_name_list[0])
    image.show()
    image = np.array(image)

    # used to pad the crops
    if design.pad_with_image_mean:
        avg_chan = np.mean(image, axis=(0,1))
    else:
        avg_chan = None
    # pad with if necessary
    frame_padded_z, npad_z = pad_frame_numpy(image, frame_sz, pos_y, pos_x, z_sz, avg_chan)
    # extract tensor of z_crops
    # print  type(design.exemplar_sz)
    z_crops = extract_crops_z_numpy(frame_padded_z, npad_z, pos_y, pos_x, z_sz, design.exemplar_sz)
    print 'the shape of the img z_crops is :' +' '+str(np.shape(z_crops))
    z_crops = np.squeeze(z_crops)
    img = Image.fromarray(z_crops.astype('uint8'), 'RGB')
    img.show()
    frame_padded_x, npad_x = pad_frame_numpy(image, frame_sz, pos_y, pos_x, x_sz2_ph, avg_chan)
    # extract tensor of x_crops (3 scales)
    x_crops = extract_crops_x_numpy(frame_padded_x, npad_x, pos_y, pos_x, x_sz0_ph, x_sz1_ph, x_sz2_ph, design.search_sz)
    print 'the shape of the img x_crops is :' +' '+str(np.shape(x_crops))
    x_crops_1 = np.squeeze(x_crops[0,:,:])
    img_1 = Image.fromarray(x_crops_1.astype('uint8'), 'RGB')
    img_1.show()
    x_crops_2 = np.squeeze(x_crops[1,:,:])
    img_2 = Image.fromarray(x_crops_2.astype('uint8'), 'RGB')
    img_2.show()
    x_crops_3 = np.squeeze(x_crops[2,:,:])
    img_3 = Image.fromarray(x_crops_3.astype('uint8'), 'RGB')
    img_3.show()

def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    print 'number of frames is :' + str(n_frames)
    print 'length of groundtruth is :' + str(len(gt))
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames

if __name__ == '__main__':
    sys.exit(main())