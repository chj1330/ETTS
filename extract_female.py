spk_list = ['ema', 'emb', 'emc', 'emd', 'eme']
def read_txt(file_list):
    spec = []
    ling = []
    mel = []
    fm = []
    spkl = []
    with open(file_list) as f:
        for line in f:
            fn = line.strip().split('|')
            for spk in spk_list:
                if fn[0].count(spk) > 0 :
                    if fn[0].count('lmy') > 0 :
                        spec.append(fn[0])
                        ling.append(fn[1])
                        mel.append(fn[2])
                        fm.append(fn[3])
                    else:
                        spec.append(fn[0])
                        ling.append(fn[1])
                        mel.append(fn[2])
                        fm.append(fn[3])

    return spec, ling, mel, fm, spkl

spec, ling, mel, fm, spk = read_txt('./all.txt')
with open('./all_emo.txt', 'w') as f:
    for idx in range(len(spec)):
        f.write("%s|%s|%s|%s\n" % (spec[idx], ling[idx], mel[idx], fm[idx]))