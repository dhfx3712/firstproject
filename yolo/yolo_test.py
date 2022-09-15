import torch

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# im = 'https://ultralytics.com/images/zidane.jpg'

model = torch.hub.load('ultralytics/yolov5', 'custom','/Users/admin/data/test_project/yolov5/yolov5s.pt')
im = '/Users/admin/data/test_project/coco128/images/train2017/000000000036.jpg'
# im = '/Users/admin/data/test_project/yolov5/data/images/zidane.jpg'
# im = '/Users/admin/Downloads/1115.jpeg'
# print (model)
print (f'im --- {im}')
# Inference
results = model(im)
# print (f'results -- {type(results)},{results.__dict__}')
print (results.print())

print (results.pandas().xyxy[0])#models.common.Detections类中pandas方法处理,在数据上增加头
print (results.pandas().xyxy)

# crops = results.crop(save=True)



