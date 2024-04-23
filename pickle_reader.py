import pickle

objects = []
with (open("saved_instances.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
    
print(len(objects))

realisations = objects[0]

print(len(realisations))

for realisation in realisations:
    print(realisation)