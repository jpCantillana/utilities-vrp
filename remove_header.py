from os import listdir

files = [f for f in listdir('./outputs9')]

for file in files:
    with open('./outputs9/'+file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open('./export9/'+file, 'w') as fout:
        fout.writelines(data[1:])