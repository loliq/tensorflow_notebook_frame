"""
     -*- coding: utf-8 -*-
    @Project: PyCharm
    @File    : iwatch.py
    @Author  : LLL
    @Site    : 
    @Email   : lilanluo@stu.xmu.edu.cn
    @Date    : 2019/5/17 10:19
    @info   :
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

def parse_single_exmp(serialized_example,is_training=True, label_num=2,
                      resize_height=224,resize_width=224):
    """
    解析tf.record
    :param serialized_example:
    :param opposite: 是否将图片取反
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据
    tf_label = tf.cast(features['label'], tf.int32)
    # TODO 图像大小不同的时候需要修改
    tf_image = tf.reshape(tf_image, [192, 192, 3])  # 设置图像的维度
    tf_image = process_image_convert_apply(tf_image, choice=True, is_training=is_training)
    tf_label = tf.one_hot(tf_label, label_num, 1, 0)
    return tf_image, tf_label

def process_image_convert_apply_crop1(image, is_training, choice = True,height=224,
                          width=224, add_image_summaries=True):
    """
    crop 后不再变形
    """
    if choice is True:
        image = -(image/255-1)
    else:
        image = image/255
    if height > 0 and width > 0:
        if is_training:
            # 数据增强，是否crop
            image = tf.image.resize_images(image, [height, width], method=np.random.randint(4))
            is_crop = np.random.randint(2)
            if is_crop:
                image = tf.slice(image, [20,20,0], [204, 204,-1])
                image = tf.image.resize_images(image, [height, width])
            # central_crop后不再进行增强
            else:
                # 决定是否进行变形增强
                deform_method = np.random.randint(3)
                if deform_method:  # 如果deform_method为0 则不进行变形
                    if deform_method is 1: # 横向拉伸
                        image = tf.slice(image, [0,0,0], [224, 192,-1])
                    elif deform_method is 2:  # 纵向拉伸
                        image = tf.slice(image, [0,0,0], [192, 224,-1])
                    image = tf.image.resize_images(image, [height, width])
                # 随机左右翻转
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image, [height, width])
    image = tf.subtract(image, 0.5)   #
    image = tf.multiply(image, 2)
    return image

def process_image_convert_apply(image, is_training, choice = True,height=224,
                          width=224, add_image_summaries=True):
    """
    crop 后继续变形
    """
                          
    if choice is True:
        image = -(image/255-1)
    else:
        image = image/255
    if height > 0 and width > 0:
        if is_training:
            image = tf.image.resize_images(image, [height, width],method = np.random.randint(4))
            # 数据增强，是否crop
            is_crop = np.random.randint(2)
            if is_crop:
                image = tf.slice(image, [20,20,0], [204, 204,-1])
            image = tf.image.resize_images(image, [height, width])
            # 决定是否进行变形增强
            deform_method = np.random.randint(3)
            if deform_method:  # 如果deform_method为0 则不进行变形
                if deform_method is 1: # 横向拉伸
                    image = tf.slice(image, [0,0,0], [224, 192,-1])
                elif deform_method is 2:  # 纵向拉伸
                    image = tf.slice(image, [0,0,0], [192, 224,-1])
                image = tf.image.resize_images(image, [height, width])
            # 随机左右翻转
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image, [height, width])
    image = tf.subtract(image, 0.5)   #
    image = tf.multiply(image, 2)
    return image

def preprocess_image(image, height, width,
                     is_training,
                     bbox=None,
                     fast_mode=True,
                     add_image_summaries=True):
    if is_training:
        return preprocess_for_train(image, height, width, bbox, fast_mode,
                                    add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(image, height, width)
def preprocess_for_train(image, height, width, bbox,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):

        # 转换图像张量类型
        if image.dtype != tf.float32:

            #  如果image.dtye=float, dtype = float,则保持不变
            #  image.dtype = interger(8位)，dtype=float 则image = image/255.0
            #   image.dtype = interger(8位)， dtype=interger， scale_in = image.dtype.max
            #   scale_out = dtype.max
            #   scale = (scale_in + 1) // (scale_out + 1)
            #   scaled = math_ops.div(image, scale)

            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1,1,4])
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes=bbox)
        distorted_image = tf.slice(image, bbox_begin, bbox_size)
        distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                                method=np.random.randint(4))
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        if add_image_summaries:
            tf.summary.image('resize_flip_image',
                             tf.expand_dims(distorted_image, 0))

        # +- 0.1 内调整亮度
        distorted_image = tf.image.random_brightness(distorted_image, 0.1)
        if add_image_summaries:
            tf.summary.image('random_brightness',
                             tf.expand_dims(distorted_image, 0))
        # 随机对比度调整
        distorted_image = tf.image.random_contrast(distorted_image, 0.8, 1.2)
        if add_image_summaries:
            tf.summary.image('random_contrast',
                             tf.expand_dims(distorted_image, 0))
        image = tf.subtract(image, 0.5)  #
        image = tf.multiply(image, 2.0)
    return  distorted_image

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0]) #去除第0维度
    # 对比度调整
    image = tf.subtract(image, 0.5)   #
    image = tf.multiply(image, 2.0)
    return image