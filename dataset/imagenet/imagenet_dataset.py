

train_datasets = ConeDateSet( args.train_dir, transforms=None)


class ConeDateSet(data.Dataset):

    def __init__(self,  lists, transforms=None, train=True, test=False):

        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]

        with open(lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []
        print ("lines:%s"%len(lines))

        for line in lines:
            # print(line)
            # a = os.path.join(root, line.split()[0])
            if os.path.exists(line.split()[0]):
                imgs.append(line.split()[0])
                # b = int(line.split()[1])
                labels.append(int(line.split()[1]))

        self.imgs = imgs
        self.labels = labels

        if transforms is None:

            self.transforms = T.Compose([
                T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.1),
                # T.Grayscale(num_output_channels=1),
                T.Resize((380, 160)),
                # T.FiveCrop(100),
                # T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                # T.Lambda(lambda tensors: torch.stack([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #                                      (t) for t in tensors])),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # T.RandomRotation(45),
                # T.CenterCrop(100),
                # T.RandomCrop(100),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert("RGB")
        # data = np.array(data)
        # data = data.astype(np.uint8)
        # data = Image.fromarray(data, 'P')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)



'''
return data,label
'''