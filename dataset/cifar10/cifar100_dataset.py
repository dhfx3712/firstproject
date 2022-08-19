
image_datasets['train'] = ImageNetTrainDataSet(os.path.join(args.data_dir, 'cifar100' ,'data'),
                                               os.path.join(args.data_dir, 'cifar100', 'label' ,'label.txt'),
                                               data_transforms['train'])






class ImageNetTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_label, data_transforms):
        # label_array = scio.loadmat(img_label)['synsets']
        label_array = [i.replace('\n','') for i in open(img_label)]
        label_array.sort()
        print(label_array)
        label_dic = {}
        for i in  range(100):
            label_dic[label_array[i]] = i
        self.img_path = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.label_dic = label_dic
        self.root_dir = root_dir
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = Image.open(data).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label

    def _make_dataset(self):
        class_to_idx = self.label_dic
        images = []
        dir = os.path.expanduser(self.root_dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    def _is_image_file(self, filename):
        """Checks if a file is an image.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)







import se_resnet
import se_resnext

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch+1)
                    resumed = True
                else:
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


