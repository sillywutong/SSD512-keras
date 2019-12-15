filename = './list_category_img.txt'
filename2 = './list_bbox.txt'
image_ids=[]
labels = []
filenames = []

with open(filename2, 'r') as bbox_file:
    with open(filename, 'r') as category_file:
        n_images = int(category_file.readline())
        for i in range(n_images):
            image_ids.append(str(i))
        next(category_file)
        category_ids = []
        for line in category_file:
            line = line.split()
            name = line[0]
            cate_id = str(line[1])
            filenames.append(name)
            category_ids.append(cate_id)
        i = 0
        next(bbox_file)
        next(bbox_file)
        for line in bbox_file:
            label_item = []#category, xmin, ymin, xmax, ymax
            line = line.split()
            gt_xmin = line[1]
            gt_ymin = line[2]
            gt_xmax = line[3]
            gt_ymax = line[4]
            label_item.append(category_ids[i])
            i+=1
            label_item.append(gt_xmin)
            label_item.append(gt_ymin)
            label_item.append(gt_xmax)
            label_item.append(gt_ymax)
            labels.append(label_item)

print(len(labels))
print(len(filenames))
print(len(image_ids))
print(labels[0])
        
        
            


