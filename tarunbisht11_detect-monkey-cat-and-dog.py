#!/usr/bin/env python
# coding: utf-8



import os
import pathlib

if "models" in pathlib.Path.cwd().parts:
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')
elif not pathlib.Path('models').exists():
    get_ipython().system('git clone --depth 1 https://github.com/tensorflow/models')




get_ipython().system('pip install pycocotools')




get_ipython().run_cell_magic('bash', '', '# Install the Object Detection API\ncd models/research/\nprotoc object_detection/protos/*.proto --python_out=.\ncp object_detection/packages/tf2/setup.py .\npython -m pip install .')




if "tensorflow-object-detection" in pathlib.Path.cwd().parts:
    while "tensorflow-object-detection" in pathlib.Path.cwd().parts:
        os.chdir('..')
elif not pathlib.Path('tensorflow-object-detection').exists():
    get_ipython().system('git clone --depth 1 https://github.com/tarun-bisht/tensorflow-object-detection.git')
pre_cwd=os.getcwd()
os.chdir("tensorflow-object-detection")




pretrained_model_url="http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"
pretrained_model_name="efficientdet_d0_coco17_tpu-32"




# pretrained_model_file=f"{pretrained_model_name}.tar.gz"
model_dir=f"../{pretrained_model_name}-theft"
pipeline_config_path=f"../pipeline.config"
output_directory= f"../{pretrained_model_name}-theft-inf"




num_steps=1000
num_eval_steps=400
num_classes=3
batch_size=16




mode="detection"
train_path="../../input/yolo-animal-detection-small/train.record"
test_path="../../input/yolo-animal-detection-small/test.record"
checkpoint_path=f"data/models/{pretrained_model_name}/checkpoint/ckpt-0"
label_path="data/labels/theft.pbtxt"




# Download a pretrained model from tensorflow model zoo
get_ipython().system('wget {pretrained_model_url}')
get_ipython().system('tar -xf {pretrained_model_name}.tar.gz')
get_ipython().system('rm {pretrained_model_name}.tar.gz')
get_ipython().system('mv {pretrained_model_name} data/models')




pipeline_file = '''# SSD with EfficientNet-b0 + BiFPN feature extractor,
# shared box predictor and focal loss (a.k.a EfficientDet-d0).
# See EfficientDet, Tan et al, https://arxiv.org/abs/1911.09070
# See Lin et al, https://arxiv.org/abs/1708.02002
# Trained on COCO, initialized from an EfficientNet-b0 checkpoint.
#
# Train on TPU-8

model {
  ssd {
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    num_classes: %d
    add_background_class: false
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    encode_background_as_zeros: true
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: [1.0, 2.0, 0.5]
        scales_per_octave: 3
      }
    }
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 512
        max_dimension: 512
        pad_to_max_dimension: true
        }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        depth: 64
        class_prediction_bias_init: -4.6
        conv_hyperparams {
          force_use_bias: true
          activation: SWISH
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            random_normal_initializer {
              stddev: 0.01
              mean: 0.0
            }
          }
          batch_norm {
            scale: true
            decay: 0.99
            epsilon: 0.001
          }
        }
        num_layers_before_predictor: 3
        kernel_size: 3
        use_depthwise: true
      }
    }
    feature_extractor {
      type: 'ssd_efficientnet-b0_bifpn_keras'
      bifpn {
        min_level: 3
        max_level: 7
        num_iterations: 3
        num_filters: 64
      }
      conv_hyperparams {
        force_use_bias: true
        activation: SWISH
        regularizer {
          l2_regularizer {
            weight: 0.00004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          scale: true,
          decay: 0.99,
          epsilon: 0.001,
        }
      }
    }
    loss {
      classification_loss {
        weighted_sigmoid_focal {
          alpha: 0.25
          gamma: 1.5
        }
      }
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    normalize_loc_loss_by_codesize: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.5
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  fine_tune_checkpoint: \"%s\"
  fine_tune_checkpoint_version: V2
  fine_tune_checkpoint_type: \"%s\"
  batch_size: %d
  sync_replicas: true
  startup_delay_steps: 0
  replicas_to_aggregate: 8
  use_bfloat16: true
  num_steps: %d
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 512
      scale_min: 0.1
      scale_max: 2.0
    }
  }
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 8e-2
          total_steps: 30000
          warmup_learning_rate: .001
          warmup_steps: 2500
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
}

train_input_reader: {
  label_map_path: \"%s\"
  tf_record_input_reader {
    input_path: \"%s\"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1;
}

eval_input_reader: {
  label_map_path: \"%s\"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: \"%s\"
  }
}''' % (num_classes, checkpoint_path, mode, batch_size, num_steps, label_path, train_path,label_path,test_path)

with open("../pipeline.config",'w') as pipe:
    pipe.write(pipeline_file)




print("Size of train record file: ",str(os.path.getsize(train_path)/1e+6)+" MB")
print("Size of test record file: ",str(os.path.getsize(test_path)/1e+6)+" MB")




get_ipython().system('python train.py     --pipeline_config_path={pipeline_config_path}     --model_dir={model_dir}     --alsologtostderr     --num_train_steps={num_steps}     --sample_1_of_n_eval_examples=1     --num_eval_steps={num_eval_steps}')




get_ipython().system('python train.py     --eval_timeout 10     --pipeline_config_path={pipeline_config_path}     --model_dir={model_dir}     --checkpoint_dir={model_dir} ')




get_ipython().system('python export_model.py     --input_type image_tensor     --pipeline_config_path {pipeline_config_path}     --trained_checkpoint_dir {model_dir}     --output_directory {output_directory}')
get_ipython().system('zip ../{pretrained_model_name}-theft-inf.zip -r {output_directory}')
get_ipython().system('rm -rf {output_directory}')




get_ipython().system('mv {pipeline_config_path} {model_dir}')
get_ipython().system('zip ../{pretrained_model_name}-theft-ckpt.zip -r {model_dir}')
get_ipython().system('rm -rf {model_dir}')




os.chdir(pre_cwd)




get_ipython().run_cell_magic('bash', '', 'rm -rf models\nrm -rf tensorflow-object-detection')

