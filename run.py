import os
import json
import cv2
from recognition.demo import *
import shutil
import numpy as np


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


if __name__ == '__main__':

    # p="C:/Users/dian3/flask-ocr/DfX_EasyOCR/"
    # p = "/home/gfarm/DfX_EasyOCR/"
    p = "/Users/shin-yujeong/Desktop/DfX_EasyOCR/"

    log = open(f'{p}log.txt', 'w')
    log.write('main\n')

    if os.listdir(p + "input_img/pregnant/"):
        print("Yes. it is a pregnant file")
        side = "pregnant"
    elif os.listdir(p + "input_img/maternity/"):
        print("Yes. it is a maternity file")
        side = "maternity"
    else:
        print("No.. there is no file")

    log.write(f'{side}\n')
    files, count = get_files(f'{p}input_img/{side}')

    log.write(f'{files}, {count}\n')
    file = cv2.imread(files[0])

    log.write(f'{file}\n')
    file = cv2.resize(file, (619, 850))
    file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    _, file = cv2.threshold(file, 155, 255, cv2.THRESH_BINARY)
    file = cv2.rectangle(file, (0, 0), (619, 850), (255, 255, 255), 15)
    file = cv2.dilate(file, np.ones((3, 3), np.uint8), iterations=1)
    file = cv2.erode(file, np.ones((3, 3), np.uint8), iterations=1)

    h_1 = 80
    h_2 = 65
    log.write(f'{h_1}, {h_2}\n')
    dict_pregnant = {'sow_no(L)': [265, 0, 500, h_1], 'sow_no(R)': [560, 0, 619, h_1],
                  'sow_birth(Y)': [89, h_1, 265, h_2], 'sow_birth(M)': [265, h_1, 442, h_2], 'sow_birth(D)': [442, h_1, 619, h_2],
                  'sow_buy(Y)': [89, h_1 + h_2, 265, h_2], 'sow_buy(M)': [265, h_1 + h_2, 442, h_2],
                  'sow_buy(D)': [442, h_1 + h_2, 619, h_2],
                  'sow_estrus(Y)': [89, h_1 + 2 * h_2, 265, h_2], 'sow_estrus(M)': [265, h_1 + 2 * h_2, 442, h_2],
                  'sow_estrus(D)': [442, h_1 + 2 * h_2, 619, h_2],
                  'sow_cross(M)': [89, h_1 + 3 * h_2, 353, h_2], 'sow_cross(D)': [353, h_1 + 3 * h_2, 619, h_2],
                  'boar_fir(L)': [89, h_1 + 4 * h_2, 231, h_2], 'boar_fir(R)': [266, h_1 + 4 * h_2, 302, h_2],
                  'boar_sec(L)': [387, h_1 + 4 * h_2, 541, h_2], 'boar_sec(R)': [580, h_1 + 4 * h_2, 619, h_2],
                  'checkdate(M)': [89, h_1 + 5 * h_2, 353, h_2], 'checkdate(D)': [353, h_1 + 5 * h_2, 619, h_2],
                  'expectdate(M)': [89, h_1 + 6 * h_2, 353, h_2], 'expectdate(D)': [353, h_1 + 6 * h_2, 619, h_2],
                  'vaccine1(M)': [89, h_1 + 7 * h_2, 196, h_2], 'vaccine1(D)': [196, h_1 + 7 * h_2, 304, h_2],
                  'vaccine2(M)': [389, h_1 + 7 * h_2, 504, h_2], 'vaccine2(D)': [504, h_1 + 7 * h_2, 619, h_2],
                  'vaccine3(M)': [89, h_1 + 8 * h_2, 196, h_2], 'vaccine3(D)': [196, h_1 + 8 * h_2, 304, h_2],
                  'vaccine4(M)': [389, h_1 + 8 * h_2, 504, h_2], 'vaccine4(D)': [504, h_1 + 8 * h_2, 619, h_2]}
    log.write(f'{dict_pregnant}\n')
    dict_maternity = {'sow_no(L)': [265, 0, 500, h_1], 'sow_no(R)': [560, 0, 619, h_1],
                  'sow_birth(Y)': [89, h_1, 265, h_2], 'sow_birth(M)': [265, h_1, 442, h_2], 'sow_birth(D)': [442, h_1, 619, h_2],
                  'sow_buy(Y)': [89, h_1 + h_2, 265, h_2], 'sow_buy(M)': [265, h_1 + h_2, 442, h_2],
                  'sow_buy(D)': [442, h_1 + h_2, 619, h_2],
                  'sow_expectdate(Y)': [89, h_1 + 2 * h_2, 265, h_2], 'sow_expectdate(M)': [265, h_1 + 2 * h_2, 442, h_2],
                  'sow_expectdate(D)': [442, h_1 + 2 * h_2, 619, h_2],
                  'sow_givebirth(M)': [89, h_1 + 3 * h_2, 353, h_2], 'sow_givebirth(D)': [353, h_1 + 3 * h_2, 619, h_2],
                  'sow_totalbaby': [89, h_1 + 4 * h_2, 232, h_2], 'sow_feedbaby': [303, h_1 + 4 * h_2, 431, h_2],
                 'sow_babyweight(L)': [501, h_1 + 4 * h_2, 559, h_2], 'sow_babyweight(R)': [559, h_1 + 4 * h_2, 619, h_2],
                  'sow_sevrerdate(M)': [89, h_1 + 5 * h_2, 353, h_2], 'sow_sevrerdate(D)': [353, h_1 + 5 * h_2, 619, h_2],
                  'sow_sevrerqty': [89, h_1 + 6 * h_2, 304, h_2],
                 'sow_sevrerweight(L)': [389, h_1 + 6 * h_2, 504, h_2], 'sow_sevrerweight(R)': [504, h_1 + 6 * h_2, 619, h_2],
                  'vaccine1(M)': [89, h_1 + 7 * h_2, 196, h_2], 'vaccine1(D)': [196, h_1 + 7 * h_2, 304, h_2],
                  'vaccine2(M)': [389, h_1 + 7 * h_2, 504, h_2], 'vaccine2(D)': [504, h_1 + 7 * h_2, 619, h_2],
                  'vaccine3(M)': [89, h_1 + 8 * h_2, 196, h_2], 'vaccine3(D)': [196, h_1 + 8 * h_2, 304, h_2],
                  'vaccine4(M)': [389, h_1 + 8 * h_2, 504, h_2], 'vaccine4(D)': [504, h_1 + 8 * h_2, 619, h_2]}
    log.write(f'{dict_maternity}\n')
    dir_num = 'cropped_imgs_num'
    if os.path.exists(f'{p}{dir_num}'):
        shutil.rmtree(f'{p}{dir_num}')
    os.makedirs(p+dir_num)

    dir_en = 'cropped_imgs_en'
    if os.path.exists(f'{p}{dir_en}'):
        shutil.rmtree(f'{p}{dir_en}')
    os.makedirs(p+dir_en)

    if side == 'pregnant':
        for k, v in dict_pregnant.items():
            v[2] = v[2] - v[0]
            x, y, w, h = v
            cropped_img = file[y:y + h, x:x + w]
            if k == 'sow_no(R)' or k == 'boar_fir(R)' or k == 'boar_sec(R)':
                cv2.imwrite(f'{p}{dir_en}/{k}.jpg', cropped_img)
            else:
                cv2.imwrite(f'{p}{dir_num}/{k}.jpg', cropped_img)
        dict_result, log = recog(dict_pregnant, log, p)
        dict_result['sow_no'] = dict_result['sow_no(L)'] +'-'+ dict_result['sow_no(R)']
        log.write(f'{dict_result["sow_no(L)"]}\n')
        log.write(f'{dict_result["sow_no(R)"]}\n')
        log.write(f'{dict_result["sow_no"]}\n')
        dict_result['boar_fir'] = dict_result['boar_fir(L)'] +'-'+ dict_result['boar_fir(R)']
        dict_result['boar_sec'] = dict_result['boar_sec(L)'] +'-'+ dict_result['boar_sec(R)']
        del (dict_result['sow_no(L)'])
        del (dict_result['sow_no(R)'])
        del (dict_result['boar_fir(L)'])
        del (dict_result['boar_fir(R)'])
        del (dict_result['boar_sec(L)'])
        del (dict_result['boar_sec(R)'])

    else:
        for k, v in dict_maternity.items():
            v[2] = v[2] - v[0]
            x, y, w, h = v
            cropped_img = file[y:y + h, x:x + w]
            if k == 'sow_no(R)':
                cv2.imwrite(f'{p}{dir_en}/{k}.jpg', cropped_img)
            else:
                cv2.imwrite(f'{p}{dir_num}/{k}.jpg', cropped_img)
        log.write(f'{os.listdir(p+dir_en)}, {os.listdir(p+dir_num)}\n')
        dict_result, log = recog(dict_maternity, log, p)
        dict_result['sow_no'] = dict_result['sow_no(L)'] +'-'+ dict_result['sow_no(R)']
        log.write(f'{dict_result["sow_no(L)"]}\n')
        log.write(f'{dict_result["sow_no(R)"]}\n')
        log.write(f'{dict_result["sow_no"]}\n')
        dict_result['sow_babyweight'] = dict_result['sow_babyweight(L)'] +'.'+ dict_result['sow_babyweight(R)']
        dict_result['sow_sevrerweight'] = dict_result['sow_sevrerweight(L)'] +'.'+ dict_result['sow_sevrerweight(R)']
        del (dict_result['sow_no(L)'])
        del (dict_result['sow_no(R)'])
        del (dict_result['sow_babyweight(L)'])
        del (dict_result['sow_babyweight(R)'])
        del (dict_result['sow_sevrerweight(L)'])
        del (dict_result['sow_sevrerweight(R)'])

    shutil.rmtree(f'{p}{dir_num}')
    shutil.rmtree(f'{p}{dir_en}')

    if side == 'maternity':
        for frontfile in os.scandir(p + "input_img/maternity/"):
            os.remove(frontfile.path)
    elif side == 'pregnant':
        for frontfile in os.scandir(p + "input_img/pregnant/"):
            os.remove(frontfile.path)

    with open(f'{p}result_json/result.json', 'w', encoding='utf-8') as f:
        json.dump(dict_result, f, indent=4, ensure_ascii=False)

    print(dict_result)

    log.close()

