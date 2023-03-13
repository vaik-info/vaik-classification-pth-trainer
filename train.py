import argparse
import os
import multiprocessing
import tqdm
from datetime import datetime
import pytz

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchmetrics
from timm.scheduler import StepLRScheduler
from torch.utils.tensorboard import SummaryWriter


from data import classification_dataset
from models import mobile_net_v2_model


def train(train_input_dir_path, valid_input_dir_path, classes_txt_path, epochs, step_size, batch_size,
          test_max_sample, image_size, output_dir_path):
    # read classes
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]

    # train dataset
    train_dataset = classification_dataset.MNISTImageDataset(train_input_dir_path, classes, batch_size * step_size,
                                                             image_size,
                                                             T.Compose([T.RandomRotation(degrees=[-0.5, 0.5]),
                                                                        T.RandomErasing(p=0.025)]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=multiprocessing.cpu_count() // 4)

    # test dataset
    test_dataset = classification_dataset.MNISTImageDataset(valid_input_dir_path, classes, test_max_sample, image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=multiprocessing.cpu_count() // 4)

    # prepare device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prepare model
    model = mobile_net_v2_model.MobileNetV2Model(len(classes))
    model.to(device)

    # prepare loss
    criterion = nn.CrossEntropyLoss()

    # prepare optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = StepLRScheduler(optimizer,
                                decay_t=epochs//4,
                                decay_rate=0.5,
                                t_in_epochs=True)

    # prepare metrics
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(classes))
    val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(classes))

    # prepare logs
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    os.makedirs(save_model_dir_path, exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(save_model_dir_path, 'logs'))

    # train
    for epoch in tqdm.tqdm(range(epochs), desc='epoch'):
        model.train()
        train_acc.reset(), val_acc.reset()
        train_epoch_loss = 0
        val_epoch_loss = 0

        for x, y in tqdm.tqdm(train_dataloader, desc='train'):
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_acc(torch.argmax(y_pred, -1).cpu(), y.cpu())
            train_epoch_loss += y_pred.shape[0] * loss.item()

        model.eval()
        for x, y in tqdm.tqdm(test_dataloader, desc='test'):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_acc(torch.argmax(y_pred, -1).cpu(), y.cpu())
                val_epoch_loss += y_pred.shape[0] * loss.item()

        train_epoch_loss = train_epoch_loss/(len(train_dataloader)*batch_size)
        val_epoch_loss = val_epoch_loss/(len(test_dataloader)*batch_size)
        print(f"Epoch{epoch+1}, Loss/train:{train_epoch_loss}, Loss/test:{val_epoch_loss},"
              f" Accuracy/train:{train_acc.compute()}, Accuracy/test:{val_acc.compute()}")
        torch.save(model.state_dict(), os.path.join(save_model_dir_path, f'epoch-{epoch}_step-{step_size}_batch-{batch_size}_train_loss-{train_epoch_loss:.4f}_val_loss-{val_epoch_loss:.4f}_train_acc-{train_acc.compute():.4f}_val_acc-{val_acc.compute():.4f}'))
        summary_writer.add_scalar("Loss/train", train_epoch_loss, epoch+1)
        summary_writer.add_scalar("Loss/test", val_epoch_loss, epoch+1)
        summary_writer.add_scalar("Accuracy/train", train_acc.compute(), epoch+1)
        summary_writer.add_scalar("Accuracy/test", val_acc.compute(), epoch+1)
        scheduler.step(epoch+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pth')
    parser.add_argument('--train_input_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/train')
    parser.add_argument('--valid_input_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/valid')
    parser.add_argument('--classes_txt_path', type=str, default='~/.vaik-mnist-classification-dataset/classes.txt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_max_sample', type=int, default=200)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-classification-pth-trainer/output_model')
    args = parser.parse_args()

    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.valid_input_dir_path = os.path.expanduser(args.valid_input_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)
    train(args.train_input_dir_path, args.valid_input_dir_path, args.classes_txt_path,
          args.epochs, args.step_size, args.batch_size, args.test_max_sample,
          (args.image_height, args.image_width), args.output_dir_path)
