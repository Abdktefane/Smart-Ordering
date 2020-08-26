# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for efficientdet_arch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf

import efficientdet_arch
import hparams_config
import utils


class EfficientDetArchTest(tf.test.TestCase):

  def build_model(self,
                  model_name,
                  isize,
                  is_training=False,
                  data_format='channels_last'):
    if isinstance(isize, int):
      isize = (isize, isize)
    if data_format == 'channels_first':
      inputs_shape = [1, 3, isize[0], isize[1]]
    else:
      inputs_shape = [1, isize[0], isize[1], 3]
    inputs = tf.ones(shape=inputs_shape, name='input', dtype=tf.float32)
    efficientdet_arch.efficientdet(
        inputs,
        model_name=model_name,
        is_training_bn=is_training,
        image_size=isize,
        data_format=data_format)
    return utils.num_params_flops(False)

  def test_efficientdet_d0(self):
    self.assertSequenceEqual((3880067, 2535456183),
                             self.build_model('efficientdet-d0', 512))

  def test_efficientdet_d0_channel_first(self):
    self.assertSequenceEqual(
        (3880067, 2534258103),
        self.build_model('efficientdet-d0', 512, data_format='channels_first'))

  def test_efficientdet_d0_511_513(self):
    self.assertSequenceEqual((3880067, 2613160475),
                             self.build_model('efficientdet-d0', (511, 513)))

  def test_efficientdet_d1(self):
    self.assertSequenceEqual((6625898, 6101276568),
                             self.build_model('efficientdet-d1', 640))

  def test_efficientdet_d1_640_1280(self):
    self.assertSequenceEqual((6625898, 12194901743),
                             self.build_model('efficientdet-d1', (640, 1280)))

  def test_efficientdet_d2(self):
    self.assertSequenceEqual((8097039, 10993616292),
                             self.build_model('efficientdet-d2', 768))

  def test_efficientdet_d3(self):
    self.assertSequenceEqual((12032296, 24882174639),
                             self.build_model('efficientdet-d3', 896))

  def test_efficientdet_d4(self):
    self.assertSequenceEqual((20723675, 55167980877),
                             self.build_model('efficientdet-d4', 1024))

  def test_efficientdet_d5(self):
    self.assertSequenceEqual((33653315, 135353202989),
                             self.build_model('efficientdet-d5', 1280))

  def test_efficientdet_d6(self):
    self.assertSequenceEqual((51871782, 225532115747),
                             self.build_model('efficientdet-d6', 1280))

  def test_efficientdet_d7(self):
    self.assertSequenceEqual((51871782, 324740293607),
                             self.build_model('efficientdet-d7', 1536))

  def test_efficientdet_lite0(self):
    self.assertSequenceEqual((3243527.0, 2504524987),
                             self.build_model('efficientdet-lite0', 512))

  def test_efficientdet_lite1(self):
    self.assertSequenceEqual((4248394.0, 3515526105),
                             self.build_model('efficientdet-lite1', 512))

  def test_efficientdet_lite2(self):
    self.assertSequenceEqual((5252429.0, 4428869862),
                             self.build_model('efficientdet-lite2', 512))

  def test_efficientdet_lite3(self):
    self.assertSequenceEqual((8350976.0, 7523573252),
                             self.build_model('efficientdet-lite3', 512))

  def test_efficientdet_lite4(self):
    self.assertSequenceEqual((15131027.0, 12977398945),
                             self.build_model('efficientdet-lite4', 512))


class EfficientDetArchPrecisionTest(tf.test.TestCase):

  def build_model(self, features, is_training, precision):

    def _model_fn(inputs):
      return efficientdet_arch.efficientdet(
          inputs,
          model_name='efficientdet-d0',
          is_training_bn=is_training,
          image_size=512)

    return utils.build_model_with_precision(precision, _model_fn, features,
                                            False)

  def test_float16(self):
    inputs = tf.ones(shape=[1, 512, 512, 3], name='input', dtype=tf.float32)
    cls_out, _ = self.build_model(inputs, True, 'mixed_float16')
    for v in tf.global_variables():
      # All variables should be float32.
      self.assertIn(v.dtype, (tf.float32, tf.dtypes.as_dtype('float32_ref')))

    for v in cls_out.values():
      self.assertIs(v.dtype, tf.float16)

  def test_bfloat16(self):
    inputs = tf.ones(shape=[1, 512, 512, 3], name='input', dtype=tf.float32)
    cls_out, _ = self.build_model(inputs, True, 'mixed_bfloat16')
    for v in tf.global_variables():
      # All variables should be float32.
      self.assertIn(v.dtype, (tf.float32, tf.dtypes.as_dtype('float32_ref')))
    for v in cls_out.values():
      self.assertEqual(v.dtype, tf.bfloat16)

  def test_float32(self):
    inputs = tf.ones(shape=[1, 512, 512, 3], name='input', dtype=tf.float32)
    cls_out, _ = self.build_model(inputs, True, 'float32')
    for v in tf.global_variables():
      # All variables should be float32.
      self.assertIn(v.dtype, (tf.float32, tf.dtypes.as_dtype('float32_ref')))
    for v in cls_out.values():
      self.assertEqual(v.dtype, tf.float32)


class BackboneTest(tf.test.TestCase):

  def test_backbone_feats(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    images = tf.ones([4, 224, 224, 3])
    feats = efficientdet_arch.build_backbone(images, config)
    self.assertEqual(list(feats.keys()), [0, 1, 2, 3, 4, 5])
    self.assertEqual(feats[0].shape, [4, 224, 224, 3])
    self.assertEqual(feats[5].shape, [4, 7, 7, 320])


class FreezeTest(tf.test.TestCase):

  def test_freeze(self):
    var_list = [
        tf.Variable(0., name='efficientnet'),
        tf.Variable(0., name='fpn_cells'),
        tf.Variable(0., name='class_net')
    ]
    freeze_var_list = efficientdet_arch.freeze_vars(var_list, None)
    self.assertEqual(len(freeze_var_list), 3)
    freeze_var_list = efficientdet_arch.freeze_vars(var_list, 'efficientnet')
    self.assertEqual(len(freeze_var_list), 2)
    freeze_var_list = efficientdet_arch.freeze_vars(var_list,
                                                    '(efficientnet|fpn_cells)')
    self.assertEqual(len(freeze_var_list), 1)


class BiFPNTest(tf.test.TestCase):

  def test_bifpn_dynamic_l3l7(self):
    p1 = efficientdet_arch.bifpn_dynamic_config(3, 7, None)
    # pyformat: disable
    self.assertEqual(
        p1.nodes,
        [
            {'feat_level': 6, 'inputs_offsets': [3, 4]},
            {'feat_level': 5, 'inputs_offsets': [2, 5]},
            {'feat_level': 4, 'inputs_offsets': [1, 6]},
            {'feat_level': 3, 'inputs_offsets': [0, 7]},
            {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},
            {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},
            {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},
            {'feat_level': 7, 'inputs_offsets': [4, 11]},
        ])
    # pyformat: enable

  def test_bifpn_dynamic_l2l7(self):
    p = efficientdet_arch.bifpn_dynamic_config(2, 7, None)

    # pyformat: disable
    self.assertEqual(
        p.nodes,
        [
            {'feat_level': 6, 'inputs_offsets': [4, 5]},
            {'feat_level': 5, 'inputs_offsets': [3, 6]},
            {'feat_level': 4, 'inputs_offsets': [2, 7]},
            {'feat_level': 3, 'inputs_offsets': [1, 8]},
            {'feat_level': 2, 'inputs_offsets': [0, 9]},
            {'feat_level': 3, 'inputs_offsets': [1, 9, 10]},
            {'feat_level': 4, 'inputs_offsets': [2, 8, 11]},
            {'feat_level': 5, 'inputs_offsets': [3, 7, 12]},
            {'feat_level': 6, 'inputs_offsets': [4, 6, 13]},
            {'feat_level': 7, 'inputs_offsets': [5, 14]},
        ])
    # pyformat: enable


class FeatureFusionTest(tf.test.TestCase):

  def test_sum(self):
    tf.disable_eager_execution()
    nodes = tf.constant([1, 3])
    nodes2 = tf.constant([1, 3])
    fused = efficientdet_arch.fuse_features([nodes, nodes2], 'sum')
    self.assertAllCloseAccordingToType(fused, [2, 6])

  def test_attn(self):
    nodes = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    fused = efficientdet_arch.fuse_features([nodes, nodes2], 'attn')

    with self.cached_session() as sess:
      # initialize weights
      sess.run(tf.global_variables_initializer())

    self.assertAllCloseAccordingToType(fused, [1.0, 3.0])

  def test_fastattn(self):
    nodes = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    fused = efficientdet_arch.fuse_features([nodes, nodes2], 'fastattn')

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())

    self.assertAllCloseAccordingToType(fused, [0.99995, 2.99985])

  def test_channel_attn(self):
    nodes = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    fused = efficientdet_arch.fuse_features([nodes, nodes2], 'channel_attn')

    with self.cached_session() as sess:
      # initialize weights
      sess.run(tf.global_variables_initializer())

    self.assertAllCloseAccordingToType(fused, [1.0, 3.0])

  def test_channel_fastattn(self):
    nodes = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    fused = efficientdet_arch.fuse_features([nodes, nodes2], 'channel_fastattn')

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())

    self.assertAllCloseAccordingToType(fused, [0.99995, 2.99985])


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.disable_eager_execution()
  tf.test.main()
