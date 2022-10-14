import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model_num, Model_en

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    if 'CTC' in opt['Prediction']:
        converter_num = CTCLabelConverter(opt['character_num'])
        converter_en = CTCLabelConverter(opt['character_en'])
    else:
        converter_num = AttnLabelConverter(opt['character_num'])
        converter_en = AttnLabelConverter(opt['character_en'])
    opt['num_class_num'] = len(converter_num.character)
    opt['num_class_en'] = len(converter_en.character)

    if opt['rgb']:
        input_channel = 3
    model_num = Model_num(opt)
    model_num = torch.nn.DataParallel(model_num).to(device)

    model_en = Model_en(opt)
    model_en = torch.nn.DataParallel(model_en).to(device)

    # load model
    model_num.load_state_dict(torch.load(opt['saved_model_num'], map_location=device))
    model_en.load_state_dict(torch.load(opt['saved_model_en'], map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=opt['PAD'])
    demo_data_num = RawDataset(root=opt['image_folder_num'], opt=opt)  # use RawDataset
    demo_loader_num = torch.utils.data.DataLoader(
        demo_data_num, batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=int(opt['workers']),
        collate_fn=AlignCollate_demo, pin_memory=True)

    demo_data_en = RawDataset(root=opt['image_folder_en'], opt=opt)  # use RawDataset
    demo_loader_en = torch.utils.data.DataLoader(
        demo_data_en, batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=int(opt['workers']),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model_num.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader_num:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt['batch_max_length']] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt['batch_max_length'] + 1).fill_(0).to(device)

            if 'CTC' in opt['Prediction']:
                preds = model_num(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter_num.decode(preds_index, preds_size)

            else:
                preds = model_num(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter_num.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt['Prediction']:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                img_name = img_name.split('/')[-1]
                if img_name.count(".") == 1:
                    name = img_name.split('.')[0]
                else:
                    for k in range(len(img_name) - 1, 0, -1):
                        if img_name[k] == '.':
                            name = img_name[:k]
                            break

                opt['dict'][name] = pred
                print(f'{name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

    model_en.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader_en:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt['batch_max_length']] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt['batch_max_length'] + 1).fill_(0).to(device)

            if 'CTC' in opt['Prediction']:
                preds = model_en(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter_en.decode(preds_index, preds_size)

            else:
                preds = model_en(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter_en.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt['Prediction']:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                img_name = img_name.split('/')[-1]
                if img_name.count(".") == 1:
                    name = img_name.split('.')[0]
                else:
                    for k in range(len(img_name) - 1, 0, -1):
                        if img_name[k] == '.':
                            name = img_name[:k]
                            break

                opt['dict'][name] = pred
                print(f'{name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
    return opt['dict']


def recog(dict):
    opt = {
        'saved_model_num': 'recognition/custom_pth/ocr_num.pth',
        'saved_model_en': 'recognition/custom_pth/ocr_en.pth',
        'image_folder_num': 'cropped_imgs_num',
        'image_folder_en': 'cropped_imgs_en',
        'dict': dict,
        'workers': 4,
        'batch_size': 192,
        'batch_max_length': 25,
        'imgH': 64,
        'imgW': 100,
        'rgb': False,
        'character_num': '0123456789',
        'character_en': 'ABCDEFGHIZKLMNOPQRSTUVWXYJabcdefghijklmnopqrstuvwxyz',
        'sensitive': True,
        'PAD': False,
        'Transformation': 'None',
        'FeatureExtraction': 'VGG',
        'SequenceModeling': 'BiLSTM',
        'Prediction': 'CTC',
        'num_fiducial': 20,
        'input_channel': 1,
        'output_channel': 512,
        'hidden_size': 256
    }

    """ vocab / character number configuration """
    if opt['sensitive']:
        # opt['character_num'] = string.printable[:-6]  # same with ASTER setting (use 94 char).
        opt['character_en'] = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt['num_gpu'] = torch.cuda.device_count()

    return demo(opt)
