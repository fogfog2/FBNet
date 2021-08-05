
import os
import cv2

def search(dirname):
    
    dirnames = os.listdir(dirname)
    dirnames = sorted(dirnames)
    dirnames=dirnames[0:400]
    for in_dir in dirnames:

        lists_ = os.path.join(dirname, in_dir , 'frames')
        lists = os.path.join('clips/', in_dir , 'frames')
        filenames = sorted(os.listdir(lists_))
        
        #for i in range(1,299):
        for path in filenames:

            full_filename = os.path.join(lists_, path)            
            img = cv2.imread(full_filename)
            img2 = cv2.resize(img, (256,256))
            cv2.imwrite(full_filename,img2)
            print(full_filename)

search("./data/datasets/custom/clips/")