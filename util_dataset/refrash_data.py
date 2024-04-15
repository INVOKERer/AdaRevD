import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import tqdm
import shutil


check_root = r'/home/ubuntu/Data1/mxt/DeFocus_HCSaveImages/train/sharp' # 106-48t/personal_data/mxt/Datasets/AutoFocus/20211101 Data1/mxt/AutoFocus/20211101
move_dir = r'/home/ubuntu/Data1/mxt/DeFocus_HCSaveImages/train/blur'
# print(check_root)
print(check_root)
images_check_list = glob.glob(check_root+'/*')
move_list = glob.glob(move_dir+'/*')
images_check = []
for x in images_check_list:
    x = os.path.basename(x)
    file_name_ = x.split('_')
    file_name_x = file_name_[0]
    for x in file_name_[1:-1]:
        file_name_x = file_name_x + '_' + x
    images_check.append(file_name_x)
# print(images_check)
print(len(images_check))
# k=0
for i in tqdm.tqdm(move_list):
    y = os.path.basename(i)
    file_name_ = y.split('_')
    file_name_y = file_name_[0]
    for y in file_name_[1:-1]:
        file_name_y = file_name_y + '_' + y
    if file_name_y not in images_check:
        print(file_name_y, i)
        # ml1 = 'sudo rm ' + i
        os.chmod(i, 0o777)
        os.remove(i)
        # k += 1
