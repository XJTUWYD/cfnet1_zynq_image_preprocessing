from __future__ import division
import numpy as np
import tensorflow as tf
from PIL import Image
import functools
import math 


def resize_images(images, size, resample):
    '''Alternative to tf.image.resize_images that uses PIL.'''
    fn = functools.partial(_resize_images, size=size, resample=resample)
    return tf.py_func(fn, [images], images.dtype)


def _resize_images(x, size, resample):
    # TODO: Use tf.map_fn?
    if len(x.shape) == 3:
        return _resize_image(x, size, resample)
    assert len(x.shape) == 4
    y = []
    for i in range(x.shape[0]):
        y.append(_resize_image(x[i]))
    y = np.stack(y, axis=0)
    return y


def _resize_image(x, size, resample):
    assert len(x.shape) == 3
    y = []
    for j in range(x.shape[2]):
        f = x[:, :, j]
        f = Image.fromarray(f)
        f = f.resize((size[1], size[0]), resample=resample)
        f = np.array(f)
        y.append(f)
    return np.stack(y, axis=2)


def pad_frame_tensorflow(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c), tf.int32))
    ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c), tf.int32))
    xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c), tf.int32) - frame_sz[1])
    ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c), tf.int32) - frame_sz[0])
    npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])
    paddings = [[npad, npad], [npad, npad], [0, 0]]
    im_padded = im
    if avg_chan is not None:
        im_padded = im_padded - avg_chan
    im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
    if avg_chan is not None:
        im_padded = im_padded + avg_chan
    return im_padded, npad


def pad_frame_numpy(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad_1 = np.round(pos_x - c)
    xleft_pad_1 = xleft_pad_1.astype(np.int32)
    xleft_pad = maximum_fun(0, -xleft_pad_1)
    xright_pad_1 = np.round(pos_x + c)
    xright_pad_1 = xright_pad_1.astype(np.int32)
    xright_pad = maximum_fun(0, -xright_pad_1) - frame_sz[1]
    ybottom_pad_1 = np.round(pos_y + c)
    ybottom_pad_1 = ybottom_pad_1.astype(np.int32)
    ybottom_pad = maximum_fun(0, -ybottom_pad_1) - frame_sz[0]
    ytop_pad_1 = np.round(pos_y - c)
    ytop_pad_1 = ytop_pad_1.astype(np.int32)
    ytop_pad = maximum_fun(0, -ytop_pad_1)
    npad = np.max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])
    paddings = [[npad, npad], [npad, npad], [0, 0]]
    im_padded = im
    if avg_chan is not None:
        im_padded = im_padded - avg_chan
    im_padded = np.pad(im_padded, paddings, mode='constant')
    if avg_chan is not None:
        im_padded = im_padded + avg_chan
    return im_padded, npad

def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    c = sz_src / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + tf.cast(tf.round(pos_x - c), tf.int32)
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tr_y = npad + tf.cast(tf.round(pos_y - c), tf.int32)
    width = tf.round(pos_x + c) - tf.round(pos_x - c)
    height = tf.round(pos_y + c) - tf.round(pos_y - c)
    crop = tf.image.crop_to_bounding_box(im,
                                         tf.cast(tr_y, tf.int32),
                                         tf.cast(tr_x, tf.int32),
                                         tf.cast(height, tf.int32),
                                         tf.cast(width, tf.int32))
    crop = tf.image.resize_images(crop, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crops = tf.expand_dims(crop, axis=0)
    return crops

#BILINEAR for image after cropped
def resize_image(img,m,n):  
    height,width,channels =img.shape  
    emptyImage=np.zeros((m,n,channels),np.uint8)  
    value=[0,0,0]  
    sh=m/height  
    sw=n/width  
    for i in range(m):  
        for j in range(n):  
            x = i/sh  
            y = j/sw  
            p=(i+0.0)/sh-x  
            q=(j+0.0)/sw-y  
            x=int(x)-1  
            y=int(y)-1  
            for k in range(3):  
                if x+1<m and y+1<n:  
                    value[k]=int(img[x,y][k]*(1-p)*(1-q)+img[x,y+1][k]*q*(1-p)+img[x+1,y][k]*(1-q)*p+img[x+1,y+1][k]*p*q)  
            emptyImage[i, j] = (value[0], value[1], value[2])  
    return emptyImage 
        

def extract_crops_z_numpy(im, npad, pos_x, pos_y, sz_src, sz_dst):
    c = sz_src / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + np.round(pos_x - c)
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tr_y = npad + np.round(pos_y - c)
    width = np.round(pos_x + c) - np.round(pos_x - c)
    height = np.round(pos_y + c) - np.round(pos_y - c)
    print 'the target positon is ' + str(tr_x)+' '+str(tr_y)+' '+str(width)+' '+str(height)
    crop = crop_to_bounding_box(im, tr_x, tr_y, height, width)  
    crop = resize_image(crop, sz_dst, sz_dst)
    crops = np.expand_dims(crop, axis=0)
    return crops


def crop_to_bounding_box(im, xmin, ymin, height, width):
    xmin = xmin.astype(np.int32)
    ymin = ymin.astype(np.int32)
    height = height.astype(np.int32)
    width = width.astype(np.int32)
    half_height = np.round(height / 2)
    half_height = half_height.astype(np.int32)
    half_width = np.round(width / 2)
    half_width = half_width.astype(np.int32)
    im = im[xmin:(xmin+width), ymin:(ymin+height) ,:]
    return im


def extract_crops_x_numpy(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    # take center of the biggest scaled source patch
    c = sz_src2 / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + np.round(pos_x - c)
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    tr_y = npad + np.round(pos_y - c)
    width = np.round(pos_x + c) - np.round(pos_x - c)
    height = np.round(pos_y + c) - np.round(pos_y - c)
    print 'data to crop bounding box:'+ str(tr_x)+str(tr_y)+str(width)+str(height)
    search_area =crop_to_bounding_box(im, tr_x, tr_y, height, width) 
    # TODO: Use computed width and height here?
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2

    crop_s0 = crop_to_bounding_box(search_area, offset_s0, offset_s0, np.round(sz_src0),np.round(sz_src0))
    crop_s0 = resize_image(crop_s0, sz_dst, sz_dst)
    crop_s1 = crop_to_bounding_box(search_area, offset_s1, offset_s1, np.round(sz_src1),np.round(sz_src1))
    crop_s1 = resize_image(crop_s1, sz_dst, sz_dst)
    crop_s2 = resize_image(search_area, sz_dst, sz_dst)
    crops = np.stack([crop_s0, crop_s1, crop_s2])
    return crops


def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    # take center of the biggest scaled source patch
    c = sz_src2 / 2
    # get top-right corner of bbox and consider padding
    tr_x = npad + tf.cast(tf.round(pos_x - c), tf.int32)
    tr_y = npad + tf.cast(tf.round(pos_y - c), tf.int32)
    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = tf.round(pos_x + c) - tf.round(pos_x - c)
    height = tf.round(pos_y + c) - tf.round(pos_y - c)
    search_area = tf.image.crop_to_bounding_box(im,
                                                tf.cast(tr_y, tf.int32),
                                                tf.cast(tr_x, tf.int32),
                                                tf.cast(height, tf.int32),
                                                tf.cast(width, tf.int32))
    # TODO: Use computed width and height here?
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2

    crop_s0 = tf.image.crop_to_bounding_box(search_area,
                                            tf.cast(offset_s0, tf.int32),
                                            tf.cast(offset_s0, tf.int32),
                                            tf.cast(tf.round(sz_src0), tf.int32),
                                            tf.cast(tf.round(sz_src0), tf.int32))
    crop_s0 = tf.image.resize_images(crop_s0, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crop_s1 = tf.image.crop_to_bounding_box(search_area,
                                            tf.cast(offset_s1, tf.int32),
                                            tf.cast(offset_s1, tf.int32),
                                            tf.cast(tf.round(sz_src1), tf.int32),
                                            tf.cast(tf.round(sz_src1), tf.int32))
    crop_s1 = tf.image.resize_images(crop_s1, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crop_s2 = tf.image.resize_images(search_area, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    crops = tf.stack([crop_s0, crop_s1, crop_s2])
    return crops

# Can't manage to use tf.crop_and_resize, which would be ideal!
# im:  A 4-D tensor of shape [batch, image_height, image_width, depth]
# boxes: the i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is
# specified in normalized coordinates [y1, x1, y2, x2]
# box_ind: specify image to which each box refers to
# crop = tf.image.crop_and_resize(im, boxes, box_ind, sz_dst)

def maximum_fun(a,b):
    if a>=b:
        return a
    else:
        return b