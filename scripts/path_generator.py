
import os

def search(dirname):
    f = open("train_files.txt",'a')
   # tt =f.readlines()
    dirnames = os.listdir(dirname)
    dirnames = sorted(dirnames)
    dirnames=dirnames[0:400]
    for in_dir in dirnames:

        lists_ = os.path.join(dirname, in_dir , 'frames')
        lists = os.path.join('clips/', in_dir , 'frames')
        filenames = sorted(os.listdir(lists_))
        
        for i in range(1,299):
            full_filename = os.path.join(lists, filenames[i])            
            f.write(full_filename+'\n')
            print(full_filename)

    f.close()

search("./data/datasets/custom/clips/")