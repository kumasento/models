#!/bin/bash

MODEL_FILE=$1
TF_ROOT=/home/rz3515/projects/tensorflow
TRANSFORM_GRAPH=$TF_ROOT/bazel-bin/tensorflow/tools/graph_transforms/transform_graph
OUTPUTS=$2

$TRANSFORM_GRAPH \
  --in_graph=$MODEL_FILE \
  --outputs="MobilenetV1/Predictions/Softmax" \
  --out_graph="frozen.pb" \
  --transforms='add_default_attributes strip_unused_nodes(type=float, shape="?,224,224,3")
    remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)
    fold_batch_norms fold_old_batch_norms quantize_weights quantize_nodes
    strip_unused_nodes sort_by_execution_order'
