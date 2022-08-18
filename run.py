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
    h = 78

    if h - a <= tl[1] and 2 * h + a >= bl[1]:   # 1행
        if 235 - a <= tl[0] and 2 * 235 + a >= tr[0]:
            return "모돈번호"
        elif 3 * 235 - a <= tl[0] and 4 * 235 + a >= tr[0]:
            return "웅돈번호"
        else:
            pass

    elif 2 * h - a <= tl[1] and 3 * h + a >= bl[1]:  # 2행
        if 104 - a <= tl[0] and 104 + 203 + a >= tr[0]:
            return "출생일(년)"
        elif 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "출생일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "출생일(일)"
        else:
            pass

    elif 3 * h - a <= tl[1] and 4 * h + a >= bl[1]:  # 3행
        if 104 - a <= tl[0] and 104 + 203 + a >= tr[0]:
            return "구입일(년)"
        elif 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "구입일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "구입일(일)"
        else:
            pass

    elif 4 * h - a <= tl[1] and 5 * h + a >= bl[1]:  # 4행
        if 104 - a <= tl[0] and 104 + 203 + a >= tr[0]:
            return "초발정일(년)"
        elif 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "초발정일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "초발정일(일)"
        else:
            pass

    elif 5 * h - a <= tl[1] and 6 * h + a >= bl[1]:  # 5행
        if 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "1차 교배일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "1차 교배일(일)"
        else:
            pass

    elif 6 * h - a <= tl[1] and 7 * h + a >= bl[1]:  # 6행
        if 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "2차 교배일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "2차 교배일(일)"
        else:
            pass

    elif 7 * h - a <= tl[1] and 8 * h + a >= bl[1]:  # 7행
        if 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "재발확인일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "재발확인일(일)"
        else:
            pass

    elif 8 * h - a <= tl[1] and 9 * h + a >= bl[1]:  # 8행
        if 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "분만예정일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "분만예정일(일)"
        else:
            pass

    elif 9 * h - a <= tl[1] and 10 * h + a >= bl[1]:    # 9행
        if 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "분만일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "분만일(일)"
        else:
            pass

    elif 10 * h - a <= tl[1] and 11 * h + a >= bl[1]:   # 10행
        if 104 - a <= tl[0] and 104 + 209 - 52 >= tr[0]:
            return "총산자수"
        elif (2 * 104) + 209 - a <= tl[0] and (2 * 104) + (2 * 209) - 52 >= tr[0]:
            return "포유개시 두수"
        elif (3 * 104) + (2 * 209) - a <= tl[0] and (3 * 104) + (3 * 209) - 35 >= tr[0]:
            return "생시체중"
        else:
            pass

    elif 11 * h - a <= tl[1] and 12 * h + a >= bl[1]:   # 11행
        if 104 + 203 + 76 - a <= tl[0] and 104 + (2 * 203) + 76 + a >= tr[0]:
            return "이유일(월)"
        elif 104 + (2 * 203) + (2 * 76) - a <= tl[0] and 104 + (3 * 203) + (2 * 76) + a >= tr[0]:
            return "이유일(일)"
        else:
            pass

    elif 12 * h - a <= tl[1] and 13 * h + a >= bl[1]:   # 12행
        if 155 - a <= tl[0] and 155 + 313 - 52 >= tr[0]:
            return "이유두수"
        elif (2 * 155) + 313 - a <= tl[0] and (2 * 155) + (2 * 313) - 35 >= tr[0]:
            return "이유체중"
        else:
            pass

    elif 13 * h - a <= tl[1] and 14 * h + a >= bl[1]:   # 13행
        if 94 - a <= tl[0] and 94 + 112 + a >= tr[0]:
            return "백신1(월)"
        elif 94 + 112 + 76 - a <= tl[0] and 94 + (2 * 112) + 76 + a >= tr[0]:
            return "백신1(일)"
        elif (2 * 94) + (2 * 112) + (2 * 76) - a <= tl[0] and (2 * 94) + (3 * 112) + (2 * 76) + a >= tr[0]:
            return "백신2(월)"
        elif (2 * 94) + (3 * 112) + (3 * 76) - a <= tl[0] and (2 * 94) + (4 * 112) + (3 * 76) + a >= tr[0]:
            return "백신2(일)"
        else:
            pass

    elif 14 * h - a <= tl[1] and 15 * h + a >= bl[1]:   # 14행
        if 94 - a <= tl[0] and 94 + 112 + a >= tr[0]:
            return "백신3(월)"
        elif 94 + 112 + 76 - a <= tl[0] and 94 + (2 * 112) + 76 + a >= tr[0]:
            return "백신3(일)"
        elif (2 * 94) + (2 * 112) + (2 * 76) - a <= tl[0] and (2 * 94) + (3 * 112) + (2 * 76) + a >= tr[0]:
            return "백신4(월)"
        elif (2 * 94) + (3 * 112) + (3 * 76) - a <= tl[0] and (2 * 94) + (4 * 112) + (3 * 76) + a >= tr[0]:
            return "백신4(일)"
        else:
            pass

    elif 15 * h - a <= tl[1]:   # 15행
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
        file = file.resize((940, 1330))
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





