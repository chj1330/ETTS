import random
import shutil
from os.path import join
train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05
def read_txt(file_list):
    list = []

    idx = 0
    with open(file_list) as f:
        for line in f:
            fn = line.split()
            list.append(fn[0])
            idx += 1
    return list


def make_list(MEL_DIR):
    result = read_txt(join(MEL_DIR, 'all.txt'))
    #shutil.copy2(join(MEL_DIR, 'train.txt'), join(MEL_DIR, 'all.txt'))

    train_size = int(len(result) * train_ratio)
    valid_size = int(len(result) * valid_ratio)

    result.sort()
    random.Random(4).shuffle(result)

    train_list = result[:train_size]
    valid_list = result[train_size:train_size+valid_size]
    test_list = result[train_size+valid_size:]


    train_list.sort()
    valid_list.sort()
    test_list.sort()
    with open(join(MEL_DIR, 'train.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % (item))
    with open(join(MEL_DIR, 'valid.txt'), 'w') as f:
        for item in valid_list:
            f.write("%s\n" % (item))
    with open(join(MEL_DIR, 'test.txt'), 'w') as f:
        for item in test_list:
            f.write("%s\n" % (item))



if __name__ == '__main__':
    DATA_ROOT = '../data'

    MEL_DIR = join(DATA_ROOT, 'feat')
    make_list(MEL_DIR)