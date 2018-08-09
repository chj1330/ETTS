import os

wav_dir = '../emotion/female'
file_list = os.listdir(wav_dir)
file_list.sort()

with open(wav_dir + '/train.txt', 'w') as f:
    for idx in range(len(file_list)):
        f.write("%s\n" % (file_list[idx]))
