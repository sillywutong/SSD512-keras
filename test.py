import numpy as np

filename = './list_category_img.txt'
filename2 = './list_bbox.txt'
partition = './list_eval_partition.txt'
image_ids=[]
labels = []
filenames = []
indices = []
with open(partition,'r') as partition_file:
    next(partition_file)
    next(partition_file)
    i=0
    for line in partition_file:
        line=line.split()
        if line[1]=='train':
            indices.append(i)
            image_ids.append(str(i))
            filenames.append(line[0])
        i+=1
print(len(image_ids))
print(image_ids[0])
print(filenames[0])
with open(filename2, 'r') as bbox_file:
    with open(filename, 'r') as category_file:
        next(category_file)
        next(category_file)
        category_ids = []
        for line in category_file:
            line = line.split()
            cate_id = str(line[1])
            category_ids.append(cate_id)
        j = 0
        next(bbox_file)
        next(bbox_file)
        for line in bbox_file:
            label_item = []#category, xmin, ymin, xmax, ymax
            line = line.split()
            gt_xmin = line[1]
            gt_ymin = line[2]
            gt_xmax = line[3]
            gt_ymax = line[4]
            label_item.append(category_ids[j])
            j+=1
            label_item.append(gt_xmin)
            label_item.append(gt_ymin)
            label_item.append(gt_xmax)
            label_item.append(gt_ymax)
            labels.append(label_item)
labels = np.array(labels)
labels = labels[indices]
labels = list(labels)
print(labels[0])
print(len(labels))
print(len(filenames))
print(len(image_ids))

        
        
            


