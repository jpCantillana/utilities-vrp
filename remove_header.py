from os import listdir

files = [f for f in listdir('./outputs10_pdp')]

for file in files:
    with open('./outputs10_pdp/'+file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open('./export10/'+file[:-3], 'w') as fout:
        # fout.writelines(data[1:])
        fout.writelines(data[2:])