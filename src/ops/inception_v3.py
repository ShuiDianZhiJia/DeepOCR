# -*- coding:utf-8 -*-
"""
@File      : inception_v3.py
@Software  : OCR
@Time      : 2018/3/31 12:09
@Author    : yubb
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tensorflow.contrib.slim as slim

IMAGE_SIZE = 35
NUM_CHANNELS = 1

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(scope, 'InceptionV3', [inputs]):
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                stride=1,
                padding='VALID'):
            end_point = 'Conv2d_1a_1x1'
            net = layers.conv2d(inputs, depth(192), [1, 1], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # # 299 x 299 x 3
            # end_point = 'Conv2d_1a_3x3'
            # net = layers.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 149 x 149 x 32
            # end_point = 'Conv2d_2a_3x3'
            # net = layers.conv2d(net, depth(32), [3, 3], scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 147 x 147 x 32
            # end_point = 'Conv2d_2b_3x3'
            # net = layers.conv2d(
            #     net, depth(64), [3, 3], padding='SAME', scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 147 x 147 x 64
            # end_point = 'MaxPool_3a_3x3'
            # net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 73 x 73 x 64
            # end_point = 'Conv2d_3b_1x1'
            # net = layers.conv2d(net, depth(80), [1, 1], scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 73 x 73 x 80.
            # end_point = 'Conv2d_4a_3x3'
            # net = layers.conv2d(net, depth(192), [3, 3], scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 71 x 71 x 192.
            # end_point = 'MaxPool_5a_3x3'
            # net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            # end_points[end_point] = net
            # if end_point == final_endpoint:
            #     return net, end_points
            # # 35 x 35 x 192.

            # Inception blocks
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                stride=1,
                padding='SAME'):
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net,
                        depth(384), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = layers.conv2d(
                        branch_1,
                        depth(96), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_1x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = layers.conv2d(
                        branch_0,
                        depth(320), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_3x3')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = layers.conv2d(
                        branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = layers.conv2d(
                        branch_1,
                        depth(192), [3, 3],
                        stride=2,
                        padding='VALID',
                        scope='Conv2d_1a_3x3')
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                            layers.conv2d(
                                branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                            layers.conv2d(
                                branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                            layers.conv2d(
                                branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = array_ops.concat(
                        [
                            layers.conv2d(
                                branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                            layers.conv2d(
                                branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                        ],
                        3)
                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = layers.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(inputs,
                 num_classes=128,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=layers_lib.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with variable_scope.variable_scope(
            scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope(
                [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
            net, end_points = inception_v3_base(
                inputs,
                scope=scope,
                min_depth=min_depth,
                depth_multiplier=depth_multiplier)

            # Auxiliary Head logits
            with arg_scope(
                    [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                    stride=1,
                    padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with variable_scope.variable_scope('AuxLogits'):
                    aux_logits = layers_lib.avg_pool2d(
                        aux_logits, [5, 5],
                        stride=3,
                        padding='VALID',
                        scope='AvgPool_1a_5x5')
                    aux_logits = layers.conv2d(
                        aux_logits, depth(128), [1, 1], scope='Conv2d_1b_1x1')

                    # Shape of feature map before the final layer.
                    kernel_size = _reduced_kernel_size_for_small_input(aux_logits, [5, 5])
                    aux_logits = layers.conv2d(
                        aux_logits,
                        depth(768),
                        kernel_size,
                        weights_initializer=trunc_normal(0.01),
                        padding='VALID',
                        scope='Conv2d_2a_{}x{}'.format(*kernel_size))
                    aux_logits = layers.conv2d(
                        aux_logits,
                        num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_initializer=trunc_normal(0.001),
                        scope='Conv2d_2b_1x1')
                    if spatial_squeeze:
                        aux_logits = array_ops.squeeze(
                            aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

            # Final pooling and prediction
            with variable_scope.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
                net = layers_lib.avg_pool2d(
                    net,
                    kernel_size,
                    padding='VALID',
                    scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 2048
                net = layers_lib.dropout(
                    net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
                # 1000
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


inception_v3.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out


def inception_v3_arg_scope(weight_decay=0.00004,
                           batch_norm_var_collection='moving_vars',
                           batch_norm_decay=0.9997,
                           batch_norm_epsilon=0.001,
                           updates_collections=ops.GraphKeys.UPDATE_OPS,
                           use_fused_batchnorm=True):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': updates_collections,
        # Use fused batch norm if possible.
        'fused': use_fused_batchnorm,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    # Set weight_decay for weights in Conv and FC layers.
    with arg_scope(
            [layers.conv2d, layers_lib.fully_connected],
            weights_regularizer=regularizers.l2_regularizer(weight_decay)):
        with arg_scope(
                [layers.conv2d],
                weights_initializer=initializers.variance_scaling_initializer(),
                activation_fn=nn_ops.relu,
                normalizer_fn=layers_lib.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc


def inference(images, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_v3(images, reuse=reuse)


def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss