from easyocr.easyocr import *
import json
import cv2
from PIL import Image
import numpy as np

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


def position(bbox):  # 사이즈는 추후에 변경
    tl, tr, br, bl = bbox
    a = 30
    row = 150
    col = 245
    h = 110
    w = 358

    if row + a >= bl[1] and 3 * col - a + 3 * w <= tl[0]:
        return "모돈번호"

    elif row - a <= tl[1] and row + a + h >= bl[1]:  # 1행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "출생일(년)"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "출생일(월)"
        elif 3 * col - a + 2 * w <= tl[0] and 3 * col + a + 3 * w >= tr[0]:
            return "출생일(일)"
        else:
            pass

    elif row - a + h <= tl[1] and row + a + 2 * h >= bl[1]:  # 2행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "구입일(년)"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "구입일(월)"
        elif 3 * col - a + 2 * w <= tl[0] and 3 * col + a + 3 * w >= tr[0]:
            return "구입일(일)"
        else:
            pass

    elif row - a + 2 * h <= tl[1] and row + a + 3 * h >= bl[1]:  # 3행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "초발정일(년)"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "초발정일(월)"
        elif 3 * col - a + 2 * w <= tl[0] and 3 * col + a + 3 * w >= tr[0]:
            return "초발정일(일)"
        else:
            pass

    elif row - a + 3 * h <= tl[1] and row + a + 4 * h >= bl[1]:  # 4행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "교배일"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "분만일"
        elif 3 * col - a + 2 * w <= tl[0] and 3 * col + a + 3 * w >= tr[0]:
            return "이유일"
        else:
            pass

    elif row - a + 4 * h <= tl[1] and row + a + 5 * h >= bl[1]:  # 5행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "웅돈번호"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "총산자수"
        elif 3 * col - a + 2 * w <= tl[0] and 3 * col + a + 3 * w >= tr[0]:
            return "이유두수"
        else:
            pass

    elif row - a + 5 * h <= tl[1] and row + a + 6 * h >= bl[1]:  # 6행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "재발확인일"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "포유개시 두수"
        elif 3 * col - a + 2 * w <= tl[0] and 3 * col + a + 3 * w >= tr[0]:
            return "이유체중"
        else:
            pass

    elif row - a + 6 * h <= tl[1] and row + a + 7 * h >= bl[1]:  # 7행
        if col - a <= tl[0] and col + a + w >= tr[0]:
            return "분만예정일"
        elif 2 * col - a + w <= tl[0] and 2 * col + a + 2 * w >= tr[0]:
            return "생시체중"
        else:
            pass

    elif row - a + 7 * h <= tl[1] and col - a <= tl[0]:
        return "특이사항"
    else:
        pass


if __name__ == '__main__':

    # # Using default model
    # reader = Reader(['en'], gpu=True)

    # Using custom model
    reader = Reader(['ko', 'en'], gpu=True,
                    model_storage_directory='model',
                    user_network_directory='user_network',
                    recog_network='custom')

    files, count = get_files('input_img')

    for idx, file in enumerate(files):
        with open('json/format.json') as f:
            data = json.load(f)

        filename = os.path.basename(file)
        file = Image.open(file)
        file = file.resize((2350, 1350))
        file = np.array(file)
        result = reader.readtext(file)

        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            pos = position(bbox)
            if pos in data:
                if data[pos] != "":
                    if isinstance(data[pos], list):
                        data[pos].append(string)
                    else:
                        d = data[pos]
                        data[pos] = []
                        data[pos].append(d)
                        data[pos].append(string)

                else:
                    data[pos] = string

            ########## bbox 그리기 ################
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            cv2.rectangle(file, tl, br, (0, 255, 0), 2)
            cv2.putText(file, string, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(f"input_img_bbox/bbox_{filename}", file)

        with open(f'json/result_json/{filename[:-4]}_result.json', 'w') as f:  # json 저장
            json.dump(data, f, indent=2, ensure_ascii=False)




