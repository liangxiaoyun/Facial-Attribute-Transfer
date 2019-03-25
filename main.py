import os
import argparse
from solver import Solver
from data_loader import get_loader   ####训练时使用这个
from data_loader_test import get_loader_test#####测试时使用这个
from torch.backends import cudnn
import numpy as np
from torchvision import transforms
#import cv2
import tkinter
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename

import time

#target_label='0000000000'    #10个属性时使用该label初始值
target_label='00000'
celeba_loader = None
rafd_loader = None
config= None
celeba_loader
rafd_loader

class target_label(object):
    def __init__(self):
        # 创建主窗口,用于容纳其它组件
        self.root = tkinter.Tk()
        # 给主窗口设置标题内容
        self.root.title("脸部属性迁移")
        #
        self.path = StringVar()

        self.choose_file =tkinter.Button(self.root, text="选择图片", command=self.selectPath)

        # 创建一个输入框,并设置尺寸
        #self.labeltext = tkinter.Label(self.root,text="请输入target label（Bangs Black_Hair Blond_Hair Brown_Hair Gray_Hair Heavy_Makeup Male Mustache Pale_Skin Young）：")
        self.labeltext = tkinter.Label(self.root, text="请输入target label（黑 金 棕 male young)")
        self.ip_input = tkinter.Entry(self.root,width=30)
        #self.ip_input.grid(row=0, column=1, padx=10, pady=5)  # 设置输入框显示的位置，以及长和宽属性
        # 创建一个输入target的按钮
        self.convert_button = tkinter.Button(self.root, command=self.translation, text="开始转换")


    # 完成布局
    def gui_arrang(self):
        self.choose_file.pack()
        self.labeltext.pack()
        self.ip_input.pack()
        self.convert_button.pack()

    #获取源文件
    def selectPath(self):
        #path_ = askdirectory()
        #config.celeba_image_dir=path_
        #tkinter.Label(self.root, text=path_).pack()
        global celeba_loader, rafd_loader
        File = askopenfilename(parent=self.root, initialdir="/Users/liangxiaoyun/", title='Choose an image.')
        if config.dataset in ['CelebA', 'Both']:
            celeba_loader = get_loader_test(File,config.celeba_crop_size, config.image_size, config.batch_size,
                                       'CelebA', config.mode, config.num_workers)
        if config.dataset in ['RaFD', 'Both']:
            rafd_loader = get_loader_test(File,config.rafd_crop_size, config.image_size, config.batch_size,
                                     'RaFD', config.mode, config.num_workers)

        config.result_dir = File[:len(File) - 4] + '_res_SN.jpg'

    # 开始转换
    def translation(self):
        # 获取输入信息
        self.ip_addr = self.ip_input.get()

        global target_label
        target_label=self.ip_addr
        #self.root.quit()

        start = time.clock()
        global celeba_loader,rafd_loader
        solver = Solver(celeba_loader, rafd_loader, config, target_label)
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()

        end = time.clock()
        print("=============================")
        print(end-start)
        print("=============================")

        image = Image.open(config.result_dir)
        photo = ImageTk.PhotoImage(image)
        imgLabel = Label(self.root, image=photo)
        imgLabel.pack()
        pass

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    global celeba_loader
    global rafd_loader

    # Solver for training and testing StarGAN.
    global target_label
    if config.mode == 'test':
        # 初始化对象
        FL = target_label()
        # 进行布局
        FL.gui_arrang()
        tkinter.mainloop()

    elif config.mode == 'train':
        if config.dataset in ['CelebA', 'Both']:
            celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                       config.celeba_crop_size, config.image_size, config.batch_size,
                                       'CelebA', config.mode, config.num_workers)
        if config.dataset in ['RaFD', 'Both']:
            rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                     config.rafd_crop_size, config.image_size, config.batch_size,
                                     'RaFD', config.mode, config.num_workers)
        solver = Solver(celeba_loader, rafd_loader, config, target_label)
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=300000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/data/testimage')
    parser.add_argument('--attr_path', type=str, default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/data/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/stargan/Celeba/models/SN')
    #parser.add_argument('--model_save_dir', type=str,default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/stargan/Celeba/models/SN_10attrs')
    parser.add_argument('--sample_dir', type=str, default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/stargan/Celeba/samples')
    parser.add_argument('--result_dir', type=str, default='/Users/liangxiaoyun/Downloads/program/Face_Style_Transfer_Program/SN-stargan-demo/stargan/Celeba/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)

