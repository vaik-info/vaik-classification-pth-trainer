import argparse
import os

import torch
from models import mobile_net_v2_model
import onnx


def export(input_pth_model_path, input_image_shape, classes_num, output_onnx_model_path):
    os.makedirs(os.path.dirname(output_onnx_model_path), exist_ok=True)
    torch_model = mobile_net_v2_model.MobileNetV2Model(classes_num)
    torch_model.load_state_dict(torch.load(input_pth_model_path))

    torch_model.eval()
    dummy_input = torch.randn(1, input_image_shape[0], input_image_shape[1], input_image_shape[2], requires_grad=True)

    torch.onnx.export(torch_model, dummy_input, output_onnx_model_path, export_params=True,
                      opset_version=10, do_constant_folding=True, input_names=['modelInput'],
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    mean, std = torch_model.get_normalize_info()
    onnx_model = onnx.load(output_onnx_model_path)
    meta_mean = onnx_model.metadata_props.add()
    meta_mean.key = "normalize_mean"
    meta_mean.value = str(mean)
    meta_std = onnx_model.metadata_props.add()
    meta_std.key = "normalize_std"
    meta_std.value = str(std)
    onnx.save(onnx_model, output_onnx_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pth')
    parser.add_argument('--input_pth_model_path', type=str,
                        default='~/.vaik-classification-pth-trainer/output_model/2023-03-13-11-02-02/epoch-8_step-1000_batch-8_train_loss-0.6040_val_loss-0.3731_train_acc-0.8706_val_acc-0.9750')
    parser.add_argument('--input_image_height', type=int, default=224)
    parser.add_argument('--input_image_width', type=int, default=224)
    parser.add_argument('--input_image_ch', type=int, default=3)
    parser.add_argument('--classes_num', type=int, default=10)
    parser.add_argument('--output_onnx_model_path', type=str,
                        default='~/.vaik-classification-pth-trainer/output_model/onnx/model.onnx')
    args = parser.parse_args()

    args.input_pth_model_path = os.path.expanduser(args.input_pth_model_path)
    args.output_onnx_model_path = os.path.expanduser(args.output_onnx_model_path)

    export(args.input_pth_model_path, (args.input_image_ch, args.input_image_height, args.input_image_width),
           args.classes_num, args.output_onnx_model_path)
