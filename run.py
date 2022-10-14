from easyocr.easyocr import *
import json
import cv2
from recognition.demo import *
import shutil


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


def imwrite(dir, k, img, params=None):
    try:
        filename = f'{k}.jpg'
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(f'{dir}/{filename}', mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':

    side = "back"

    files, count = get_files(f'input_img/{side}')

    file = cv2.imread(files[0])
    file = cv2.resize(file, (619, 850))
    file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    _, file = cv2.threshold(file, 155, 255, cv2.THRESH_BINARY)

    h_1 = 80
    h_2 = 65

    dict_front = {'모돈번호(왼)': [265, 0, 500, h_1], '모돈번호(오)': [560, 0, 619, h_1],
                  '출생일(년)': [89, h_1, 265, h_2], '출생일(월)': [265, h_1, 442, h_2], '출생일(일)': [442, h_1, 619, h_2],
                  '구입일(년)': [89, h_1 + h_2, 265, h_2], '구입일(월)': [265, h_1 + h_2, 442, h_2],
                  '구입일(일)': [442, h_1 + h_2, 619, h_2],
                  '초발정일(년)': [89, h_1 + 2 * h_2, 265, h_2], '초발정일(월)': [265, h_1 + 2 * h_2, 442, h_2],
                  '초발정일(일)': [442, h_1 + 2 * h_2, 619, h_2],
                  '교배일(월)': [89, h_1 + 3 * h_2, 353, h_2], '교배일(일)': [353, h_1 + 3 * h_2, 619, h_2],
                  '1차웅돈번호(왼)': [89, h_1 + 4 * h_2, 231, h_2], '1차웅돈번호(오)': [266, h_1 + 4 * h_2, 302, h_2],
                  '2차웅돈번호(왼)': [387, h_1 + 4 * h_2, 541, h_2], '2차웅돈번호(오)': [580, h_1 + 4 * h_2, 619, h_2],
                  '재발확인일(월)': [89, h_1 + 5 * h_2, 353, h_2], '재발확인일(일)': [353, h_1 + 5 * h_2, 619, h_2],
                  '분만예정일(월)': [89, h_1 + 6 * h_2, 353, h_2], '분만예정일(일)': [353, h_1 + 6 * h_2, 619, h_2],
                  '백신1(월)': [89, h_1 + 7 * h_2, 196, h_2], '백신1(일)': [196, h_1 + 7 * h_2, 304, h_2],
                  '백신2(월)': [389, h_1 + 7 * h_2, 504, h_2], '백신2(일)': [504, h_1 + 7 * h_2, 619, h_2],
                  '백신3(월)': [89, h_1 + 8 * h_2, 196, h_2], '백신3(일)': [196, h_1 + 8 * h_2, 304, h_2],
                  '백신4(월)': [389, h_1 + 8 * h_2, 504, h_2], '백신4(일)': [504, h_1 + 8 * h_2, 619, h_2]}

    dict_back = {'모돈번호(왼)': [265, 0, 500, h_1], '모돈번호(오)': [560, 0, 619, h_1],
                  '출생일(년)': [89, h_1, 265, h_2], '출생일(월)': [265, h_1, 442, h_2], '출생일(일)': [442, h_1, 619, h_2],
                  '구입일(년)': [89, h_1 + h_2, 265, h_2], '구입일(월)': [265, h_1 + h_2, 442, h_2],
                  '구입일(일)': [442, h_1 + h_2, 619, h_2],
                  '분만예정일(년)': [89, h_1 + 2 * h_2, 265, h_2], '분만예정일(월)': [265, h_1 + 2 * h_2, 442, h_2],
                  '분만예정일(일)': [442, h_1 + 2 * h_2, 619, h_2],
                  '분만일(월)': [89, h_1 + 3 * h_2, 353, h_2], '분만일(일)': [353, h_1 + 3 * h_2, 619, h_2],
                  '총산자수': [89, h_1 + 4 * h_2, 232, h_2], '포유개시두수': [303, h_1 + 4 * h_2, 431, h_2],
                 '생시체중(왼)': [501, h_1 + 4 * h_2, 559, h_2], '생시체중(오)': [559, h_1 + 4 * h_2, 619, h_2],
                  '이유일(월)': [89, h_1 + 5 * h_2, 353, h_2], '이유일(일)': [353, h_1 + 5 * h_2, 619, h_2],
                  '이유두수': [89, h_1 + 6 * h_2, 304, h_2],
                 '이유체중(왼)': [389, h_1 + 6 * h_2, 504, h_2], '이유체중(오)': [504, h_1 + 6 * h_2, 619, h_2],
                  '백신1(월)': [89, h_1 + 7 * h_2, 196, h_2], '백신1(일)': [196, h_1 + 7 * h_2, 304, h_2],
                  '백신2(월)': [389, h_1 + 7 * h_2, 504, h_2], '백신2(일)': [504, h_1 + 7 * h_2, 619, h_2],
                  '백신3(월)': [89, h_1 + 8 * h_2, 196, h_2], '백신3(일)': [196, h_1 + 8 * h_2, 304, h_2],
                  '백신4(월)': [389, h_1 + 8 * h_2, 504, h_2], '백신4(일)': [504, h_1 + 8 * h_2, 619, h_2]}

    dir_num = 'cropped_imgs_num'
    if os.path.exists(f'{dir_num}'):
        shutil.rmtree(f'{dir_num}')
    os.makedirs(dir_num)

    dir_en = 'cropped_imgs_en'
    if os.path.exists(f'{dir_en}'):
        shutil.rmtree(f'{dir_en}')
    os.makedirs(dir_en)

    if side == 'front':
        for k, v in dict_front.items():
            v[2] = v[2] - v[0]
            x, y, w, h = v
            cropped_img = file[y:y + h, x:x + w]
            if k == '모돈번호(오)' or k == '1차웅돈번호(오)' or k == '2차웅돈번호(오)':
                imwrite(dir_en, k, cropped_img)
            else:
                imwrite(dir_num, k, cropped_img)
        dict_result = recog(dict_front)
        dict_result['모돈번호'] = dict_result['모돈번호(왼)'] +'-'+ dict_result['모돈번호(오)']
        dict_result['1차웅돈번호'] = dict_result['1차웅돈번호(왼)'] +'-'+ dict_result['1차웅돈번호(오)']
        dict_result['2차웅돈번호'] = dict_result['2차웅돈번호(왼)'] +'-'+ dict_result['2차웅돈번호(오)']
        del (dict_result['모돈번호(왼)'])
        del (dict_result['모돈번호(오)'])
        del (dict_result['1차웅돈번호(왼)'])
        del (dict_result['1차웅돈번호(오)'])
        del (dict_result['2차웅돈번호(왼)'])
        del (dict_result['2차웅돈번호(오)'])

    else:
        for k, v in dict_back.items():
            v[2] = v[2] - v[0]
            x, y, w, h = v
            cropped_img = file[y:y + h, x:x + w]
            if k == '모돈번호(오)':
                imwrite(dir_en, k, cropped_img)
            else:
                imwrite(dir_num, k, cropped_img)
        dict_result = recog(dict_back)
        dict_result['모돈번호'] = dict_result['모돈번호(왼)'] +'-'+ dict_result['모돈번호(오)']
        dict_result['생시체중'] = dict_result['생시체중(왼)'] +'.'+ dict_result['생시체중(오)']
        dict_result['이유체중'] = dict_result['이유체중(왼)'] +'.'+ dict_result['이유체중(오)']
        del (dict_result['모돈번호(왼)'])
        del (dict_result['모돈번호(오)'])
        del (dict_result['생시체중(왼)'])
        del (dict_result['생시체중(오)'])
        del (dict_result['이유체중(왼)'])
        del (dict_result['이유체중(오)'])

    shutil.rmtree(f'{dir_num}')
    shutil.rmtree(f'{dir_en}')

    with open('result_json/result.json', 'w', encoding='utf-8') as f:
        json.dump(dict_result, f, indent=4, ensure_ascii=False)

    print(dict_result)

