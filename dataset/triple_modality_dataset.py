# 파일 이름: dataset/triple_modality_dataset.py (최종 버전)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Batch
from dataset.triple_dataset_test import MolTestDataset, scaffold_split

# -------------------------------------------------------------------
# 1. 트리플 모달리티를 위한 새로운 Dataset 클래스
# -------------------------------------------------------------------
class TripleModalityDataset(Dataset):
    """
    원본 MolTestDataset을 감싸고, KV-PLM 특징 벡터를 추가로 관리하는 클래스.
    """
    def __init__(self, original_dataset: MolTestDataset, kv_features: np.ndarray = None):
        super(TripleModalityDataset, self).__init__()
        self.original_dataset = original_dataset
        self.smiles_data = self.original_dataset.smiles_data
        self.kv_features = kv_features
        
        if self.kv_features is not None:
            assert len(self.original_dataset) == len(self.kv_features), \
                "Length of original dataset and KV-PLM features must match!"

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # MolTestDataset이 반환하는 graph_data (y 포함)를 그대로 사용
        graph_data = self.original_dataset[idx] 
        smiles = self.smiles_data[idx]
        kv_feature = torch.tensor(self.kv_features[idx], dtype=torch.float)
        # label을 따로 반환하지 않음
        return graph_data, smiles, kv_feature

# -------------------------------------------------------------------
# 2. 트리플 모달리티를 위한 새로운 Collate 함수
# -------------------------------------------------------------------
def triple_modality_collate_fn(batch):
    graph_list, smiles_list, kv_features_list = zip(*batch)
    
    graph_batch = Batch.from_data_list(list(graph_list))
    kv_features_tensor = torch.stack(kv_features_list)
    
    # graph_batch.y에서 레이블을 직접 추출
    labels_tensor = graph_batch.y
    # collate_fn의 출력 형식은 유지
    return graph_batch, list(smiles_list), labels_tensor, kv_features_tensor

# -------------------------------------------------------------------
# 3. 트리플 모달리티를 위한 최종 Wrapper 클래스
# -------------------------------------------------------------------
class TripleModalityDatasetWrapper:
    def __init__(self, batch_size: int, num_workers: int, valid_size: float, test_size: float,
                 data_path: str, target: str, task: str, 
                 kv_plm_features_path: str, dataset_name: str, 
                 splitting: str = 'scaffold'):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size

        # --- [핵심] 데이터 필터링 및 KV-PLM 특징 로드 ---
        df_original = pd.read_csv(data_path)
        indices_path = f'data/{dataset_name.lower()}/valid_indices_{dataset_name}.npy'
        valid_indices = np.load(indices_path)
        
        # 원본 데이터프레임을 필터링
        df_valid = df_original.iloc[valid_indices].reset_index(drop=True)
        # 필터링된 데이터프레임을 임시 파일로 저장 (MolTestDataset이 읽도록)
        filtered_data_path = f'data/{dataset_name.lower()}/filtered_{dataset_name}.csv'
        df_valid.to_csv(filtered_data_path, index=False)

        # KV-PLM 특징 로드
        kv_features = np.load(kv_plm_features_path)

        # 1. **필터링된** CSV 파일을 사용하여 MolTestDataset 인스턴스 생성
        #    이제 MolTestDataset은 유효한 2039개의 데이터만 로드하게 됨
        full_dataset = MolTestDataset(data_path=filtered_data_path, target=target, task=task)
        
        assert len(full_dataset) == len(kv_features), "Data length mismatch after filtering!"

        # 2. Scaffold 분할 수행
        train_idx, valid_idx, test_idx = scaffold_split(full_dataset, valid_size, test_size)

        # 3. **핵심**: 원본 데이터셋과 KV-PLM 특징을 우리의 TripleModalityDataset으로 감싸줌
        #    이때 전체 데이터셋(full_dataset)을 전달
        self.full_triple_dataset = TripleModalityDataset(full_dataset, kv_features)

        # 4. 각 데이터 분할에 대한 Sampler 정의
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def get_data_loaders(self):
        # 5. 최종 데이터 로더 생성
        train_loader = DataLoader(
            self.full_triple_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
            num_workers=self.num_workers, drop_last=False, collate_fn=triple_modality_collate_fn
        )
        valid_loader = DataLoader(
            self.full_triple_dataset, batch_size=self.batch_size, sampler=self.valid_sampler,
            num_workers=self.num_workers, drop_last=False, collate_fn=triple_modality_collate_fn
        )
        test_loader = DataLoader(
            self.full_triple_dataset, batch_size=self.batch_size, sampler=self.test_sampler,
            num_workers=self.num_workers, drop_last=False, collate_fn=triple_modality_collate_fn
        )
        return train_loader, valid_loader, test_loader