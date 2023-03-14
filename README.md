# vaik-classification-pth-trainer

Train classification pth model

## train_pth.py

### Usage

```shell
pip install -r requirements.txt
python train.py --train_input_dir_path ~/.vaik-mnist-classification-dataset/train \
                --valid_input_dir_path ~/.vaik-mnist-classification-dataset/valid \
                --classes_txt_path ~/.vaik-mnist-classification-dataset/classes.txt \
                --epochs 10 \
                --step_size 2000 \
                --batch_size 8 \
                --test_max_sample 100 \
                --image_size 224 \
                --output_dir_path ~/.vaik-classification-pth-trainer/output_model
```

- train_input_dir_path & valid_input_dir_path

```shell
.
├── eight
│   ├── valid_000000024.jpg
│   ├── valid_000000034.jpg
・・・
│   └── valid_000001976.jpg
├── five
│   ├── valid_000000016.jpg
・・・
```

### Output

![Screenshot from 2023-03-14 14-46-15](https://user-images.githubusercontent.com/116471878/224907745-d66d07cc-0b3e-4170-8695-9c321fbb5a1a.png)
 
-----

## export_onnx.py

### Usage

```shell
pip install -r requirements.txt
python export_onnx.py --input_pth_model_path ~/.vaik-classification-pth-trainer/output_model/2023-03-13-11-02-02/epoch-8_step-1000_batch-8_train_loss-0.6040_val_loss-0.3731_train_acc-0.8706_val_acc-0.9750 \
                --input_image_height 224 \
                --input_image_width 224 \
                --input_image_ch 3 \
                --classes_num 10 \
                --opset_version 10 \
                --output_onnx_model_path ~/.vaik-classification-pth-trainer/output_model/onnx/model.onnx
```

### Output

- ~/.vaik-classification-pth-trainer/output_model/onnx/model.onnx
