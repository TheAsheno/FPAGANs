import torch
import torchvision
import os
import argparse
import logging

from tqdm import tqdm
from dataLoader.AgeAlexNet_data import CACD
from model.AgeAlexNet import AgeAlexNet
from utils.io import check_dir, Img_to_zero_center

parser = argparse.ArgumentParser(description='pretrain age classifier')

parser.add_argument('--learning_rate', '--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=256)
parser.add_argument('--max_epoches', type=int, help='Number of epoches to run', default=200)
parser.add_argument('--val_interval', type=int, help='Number of steps to validate', default=20000)
parser.add_argument('--save_interval', type=int, help='Number of batches to save model', default=20000)

parser.add_argument('--cuda_device', type=str, help='which device to use', default='0')
parser.add_argument('--checkpoint', type=str, help='logs and checkpoints directory', default='./checkpoint/AgeClassify')
parser.add_argument('--saved_model_folder', type=str,
                    help='the path of folder which stores the parameters file',
                    default='./checkpoint/AgeClassify/saved_parameters/')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

check_dir(args.checkpoint)
check_dir(args.saved_model_folder)

logger = logging.getLogger("Age classifer")
file_handler = logging.FileHandler(os.path.join(args.checkpoint, 'log.txt'), "w")
stdout_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.setLevel(logging.INFO)


def main():
    logger.info("Start to train:\n arguments: %s" % str(args))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((227, 227)),
        torchvision.transforms.ToTensor(),
        Img_to_zero_center()
    ])

    train_dataset = CACD("train", transforms)
    test_dataset = CACD("test", transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = AgeAlexNet().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for epoch in range(args.max_epoches):
        for idx, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()

            model.train()
            optim.zero_grad()
            output = model(img)
            loss = model.loss(output, label.long())
            loss.backward()
            optim.step()
            format_str = ('step %d/%d, cls_loss = %.3f')
            logger.info(format_str % (idx, len(train_loader), loss))

            if idx * args.batch_size % args.save_interval == 0:
                model.save_model(dir=args.saved_model_folder,
                                 filename='epoch_%d_iter_%d.pth'%(epoch, idx))
                logger.info('checkpoint has been created!')

            if idx % args.val_interval == 0:
                model.eval()
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for val_img, val_label in tqdm(test_loader):
                        val_img = val_img.cuda()
                        val_label = val_label.cuda()
                        output = model(val_img).argmax(dim=1)
                        test_correct += (output == val_label).sum().item()
                        test_total += val_img.size()[0]

                logger.info('validate has been finished!')
                format_str = ('val_acc = %.3f')
                logger.info(format_str % (test_correct / test_total))

if __name__ == '__main__':
    main()
