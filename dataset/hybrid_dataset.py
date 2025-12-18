import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Batch

# finetune.py가 사용하는 실제 파일에서 클래스와 함수를 임포트합니다.
from dataset.dataset_test import MolTestDataset, scaffold_split

# collate_fn은 그래프와 SMILES 리스트를 올바르게 배치로 만들어줍니다. (수정 불필요)
def collate_fn(batch):
    graph_list, smiles_list = zip(*batch)
    graph_batch = Batch.from_data_list(list(graph_list))
    return graph_batch, list(smiles_list)


class HybridMolDataset(Dataset):
    """
    원본 MolTestDataset을 감싸는(wrapping) 간단한 클래스.
    __getitem__ 호출 시 그래프 데이터와 해당 인덱스의 SMILES 문자열을 함께 반환합니다.
    """
    def __init__(self, original_dataset):
        super(HybridMolDataset, self).__init__()
        self.original_dataset = original_dataset
        # 원본 데이터셋이 로드한 SMILES 리스트를 직접 참조합니다.
        self.smiles_data = self.original_dataset.smiles_data

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 1. 원본 데이터셋에서 그래프 데이터를 가져옵니다.
        graph_data = self.original_dataset[idx]
        # 2. SMILES 리스트에서 해당 인덱스의 SMILES 문자열을 가져옵니다.
        smiles = self.smiles_data[idx]
        # 3. 두 데이터를 튜플로 묶어 반환합니다.
        return graph_data, smiles


class HybridDatasetWrapper:
    """
    기존 MolTestDatasetWrapper의 데이터 분할 로직을 그대로 재현하면서,
    우리의 HybridMolDataset을 사용하여 데이터 로더를 생성하는 최종 래퍼 클래스.
    """
    def __init__(self, batch_size, num_workers, valid_size, test_size, 
                 data_path, target, task, splitting):
        super(HybridDatasetWrapper, self).__init__()
        # config 파일로부터 받은 모든 인자를 저장합니다.
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.data_path = data_path
        self.target = target
        self.task = task
        self.splitting = splitting
        assert self.splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        # 1. 원본 MolTestDataset 인스턴스를 생성합니다.
        full_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)

        # 2. 원본 MolTestDatasetWrapper의 로직과 동일하게 train/valid/test 인덱스를 얻습니다.
        if self.splitting == 'random':
            num_train = len(full_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(full_dataset, self.valid_size, self.test_size)

        # 3. 각 데이터 분할에 대한 Sampler를 정의합니다.
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # 4. **핵심**: 원본 데이터셋을 우리의 HybridMolDataset으로 감싸줍니다.
        hybrid_dataset = HybridMolDataset(full_dataset)

        # 5. Hybrid 데이터셋과 Sampler를 사용하여 최종 데이터 로더를 생성합니다.
        #    반드시 collate_fn을 지정하여 (그래프, SMILES) 쌍을 올바르게 배치 처리하도록 합니다.
        train_loader = DataLoader(
            hybrid_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False, collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            hybrid_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            hybrid_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False, collate_fn=collate_fn
        )

        return train_loader, valid_loader, test_loader