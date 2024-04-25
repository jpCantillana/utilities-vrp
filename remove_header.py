from os import listdir

files = [f for f in listdir('./outputs')]

for file in files:
    with open('./outputs/'+file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open('./export/'+file, 'w') as fout:
        fout.writelines(data[1:])