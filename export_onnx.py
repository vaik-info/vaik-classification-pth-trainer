import argparse
import os

import torch
import torch.nn as nn
from models import mobile_net_v2_model


def export(input_pth_model_path, input_image_shape, classes_num, opset_version, output_onnx_model_path):
    os.makedirs(os.path.dirname(output_onnx_model_path), exist_ok=True)
    torch_model = mobile_net_v2_model.MobileNetV2Model(classes_num, preprocessing=lambda x: torch.permute(torch.div(x, 255.), (0, 3, 1, 2)))
    torch_model.mobile_net_v2.classifier.add_module('softmax', nn.Softmax(dim=1))

    torch_model.load_state_dict(torch.load(input_pth_model_path))

    torch_model.eval()
    dummy_input = torch.randn(1, input_image_shape[0], input_image_shape[1], input_image_shape[2], requires_grad=True)

    torch.onnx.export(torch_model, dummy_input, output_onnx_model_path, export_params=True,
                      opset_version=opset_version, do_constant_folding=True, input_names=['input'],
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('--input_pth_model_path', type=str,
                        default='~/.vaik-classification-pth-trainer/output_model/2023-03-13-11-02-02/epoch-8_step-1000_batch-8_train_loss-0.6040_val_loss-0.3731_train_acc-0.8706_val_acc-0.9750')
    parser.add_argument('--input_image_height', type=int, default=224)
    parser.add_argument('--input_image_width', type=int, default=224)
    parser.add_argument('--input_image_ch', type=int, default=3)
    parser.add_argument('--classes_num', type=int, default=10)
    parser.add_argument('--opset_version', type=int, default=10)
    parser.add_argument('--output_onnx_model_path', type=str,
                        default='~/.vaik-classification-pth-trainer/output_model/onnx/model.onnx')
    args = parser.parse_args()

    args.input_pth_model_path = os.path.expanduser(args.input_pth_model_path)
    args.output_onnx_model_path = os.path.expanduser(args.output_onnx_model_path)

    export(args.input_pth_model_path, (args.input_image_height, args.input_image_width, args.input_image_ch),
           args.classes_num, args.opset_version, args.output_onnx_model_path)
