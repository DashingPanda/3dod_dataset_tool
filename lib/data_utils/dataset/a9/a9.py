import torch.utils.data as data



class A9(data.Dataset):
    def __init__(self, root_dir, dataset_txt):
        self.idx_list = [x.strip() for x in open(dataset_txt).readlines()]
        print(f'self.idx_list[0]: {self.idx_list[0]}')