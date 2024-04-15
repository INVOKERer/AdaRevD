import cv2
import os


def imgResize(file_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith(('.jpg', '.png', '.bmp')):  # 判断是否以.jpg结尾
            file = file_dir + '/' + files
            imgOri = cv2.imread(file)
            h, w, _ = imgOri.shape
            size = (h*2, w*2)
            # size = (2048, 2048)
            result = cv2.resize(imgOri, size, interpolation=cv2.INTER_AREA)
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + '/' + filename + '.png'
            cv2.imwrite(outputway, result)


if __name__ == '__main__':
    folder = '/home/mxt/106-48t/personal_data/mxt/Datasets/PAN/PAN_DegradeX-airport/test/degrade_1'      # 存放图片的文件夹
    outWay = '/home/mxt/106-48t/personal_data/mxt/Datasets/PAN/PAN_DegradeX-airport/test/DRx2'  # 输出路径
    imgResize(folder, outWay)

