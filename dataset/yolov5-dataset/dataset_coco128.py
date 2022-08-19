

train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                          hyp=hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                          rect=opt.rect, rank=LOCAL_RANK, workers=workers,
                                          image_weights=opt.image_weights, quad=opt.quad,
                                          prefix=colorstr('train: '), shuffle=True)




dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                              augment=augment,  # augmentation
                              hyp=hyp,  # hyperparameters
                              rect=rect,  # rectangular batches
                              cache_images=cache,
                              single_cls=single_cls,
                              stride=int(stride),
                              pad=pad,
                              image_weights=image_weights,
                              prefix=prefix)



loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
loader(dataset,
              batch_size=batch_size,
              shuffle=shuffle and sampler is None,
              num_workers=nw,
              sampler=sampler,
              pin_memory=True,
              collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset

''''
def __len__(self):
    return len(self.img_files)

def __getitem__(self, index):
    return torch.from_numpy(img), labels_out, self.img_files[index], shapes

'''


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__() #超类，直接引用父类方法

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


#/Users/admin/opt/anaconda3/envs/openmmlab/lib/python3.7/site-packages/torch/utils/data/dataloader.py


    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


#super().__init__() 。https://blog.csdn.net/a__int__/article/details/104600972
class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

class _BaseDataLoaderIter(object):

    def __next__(self):
        data = self._next_data()


for i, (imgs, targets, paths, _) in pbar #每个批次返回一个list列表