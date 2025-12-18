import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.hybrid_dataset import HybridDatasetWrapper
from models.concatenate_model import HybridModel

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Apex 라이브러리를 조건부로 임포트합니다
apex_support = False
try:
    from apex import amp
    apex_support = True
    print("Apex found. Mixed precision training is available.")
except ImportError:
    print("Apex not found. Training in full precision (FP32).")


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./DRGS-Net.yaml', os.path.join(model_checkpoints_folder, 'config_concatenate.yaml'))


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # 로그 디렉토리 이름을 조금 더 간결하게 수정 (target 이름은 제외)
        dir_name = current_time + '_HYBRID_' + config['task_name']
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            self.criterion = nn.L1Loss() if self.config["task_name"] in ['qm7', 'qm8', 'qm9'] else nn.MSELoss()

    def _get_device(self):
        device = torch.device(self.config['gpu'] if torch.cuda.is_available() else 'cpu')
        print("Running on:", device)
        return device

    def _step(self, model, graph_data, smiles_list):
        __, pred = model(graph_data, smiles_list)
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, graph_data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            labels = graph_data.y
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(labels))
            else:
                loss = self.criterion(pred, labels)
        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for graph_batch, _ in train_loader:
                labels.append(graph_batch.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print("Labels normalized. Mean:", self.normalizer.mean.item(), "Std:", self.normalizer.std.item())

        model = HybridModel(
            molclr_config=self.config["model"],
            task_type=self.config['dataset']['task'],
            chemberta_model_name=self.config['hybrid_specific']['chemberta_model_name']
        ).to(self.device)

        self._load_molclr_pre_trained_weights(model)

        optimizer = torch.optim.Adam([
            {'params': model.molclr_model.parameters(), 'lr': float(self.config['init_base_lr'])},
            {'params': model.chemberta_model.parameters(), 'lr': float(self.config['hybrid_specific']['chemberta_lr'])},
            {'params': model.hybrid_pred_head.parameters(), 'lr': float(self.config['init_lr'])}
        ], weight_decay=float(self.config['weight_decay']))

        if apex_support and self.config['fp16_precision']:
            print("Activating mixed precision training with Apex.")
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, self.config['dataset']['target'], 'checkpoints')
        _save_config_file(model_checkpoints_folder)
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_counter}/{self.config['epochs']}")
            for bn, (graph_data, smiles_list) in enumerate(progress_bar):
                optimizer.zero_grad()
                graph_data = graph_data.to(self.device)
                
                loss = self._step(model, graph_data, smiles_list)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

                progress_bar.set_postfix({'loss': loss.item()})
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar(f'{self.config["dataset"]["target"]}/train_loss', loss, global_step=n_iter)
            
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                print(f"\n--- Validating Epoch {epoch_counter} for target: {self.config['dataset']['target']} ---")
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        print(f"** New best ROC AUC: {best_valid_cls:.4f}. Model saved! **")
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        metric_name = "MAE" if self.config["task_name"] in ['qm7', 'qm8', 'qm9'] else "RMSE"
                        print(f"** New best {metric_name}: {best_valid_rgr:.4f}. Model saved! **")
                
                self.writer.add_scalar(f'{self.config["dataset"]["target"]}/validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        print(f"\n--- Final Testing for target: {self.config['dataset']['target']} ---")
        # _test 메서드가 반환하는 최종 점수를 받아서 return
        final_test_score = self._test(model, test_loader)
        return final_test_score

    def _load_molclr_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_molclr_pre_trained_weights(state_dict)
            print("Successfully loaded pre-trained MolCLR weights.")
        except FileNotFoundError:
            print("Pre-trained MolCLR weights not found. MolCLR part will be trained from scratch.")
        return model

    def _validate(self, model, valid_loader):
        predictions, labels = [], []
        with torch.no_grad():
            model.eval()
            valid_loss, num_data = 0.0, 0
            for graph_data, smiles_list in tqdm(valid_loader, desc="Validating"):
                graph_data = graph_data.to(self.device)
                
                __, pred = model(graph_data, smiles_list)
                loss = self._step(model, graph_data, smiles_list)
                valid_loss += loss.item() * graph_data.y.size(0)
                num_data += graph_data.y.size(0)

                if self.normalizer: pred = self.normalizer.denorm(pred)
                if self.config['dataset']['task'] == 'classification': pred = F.softmax(pred, dim=-1)

                predictions.extend(pred.cpu().detach().numpy())
                labels.extend(graph_data.y.cpu().flatten().numpy())
            valid_loss /= num_data
        model.train()
        
        predictions, labels = np.array(predictions), np.array(labels)
        if self.config['dataset']['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print(f'Validation Loss: {valid_loss:.4f}, MAE: {mae:.4f}')
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print(f'Validation Loss: {valid_loss:.4f}, RMSE: {rmse:.4f}')
                return valid_loss, rmse
        elif self.config['dataset']['task'] == 'classification':
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print(f'Validation Loss: {valid_loss:.4f}, ROC AUC: {roc_auc:.4f}')
            return valid_loss, roc_auc
    
    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, self.config['dataset']['target'], 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded best trained model for testing.")

        predictions, labels = [], []
        with torch.no_grad():
            model.eval()
            test_loss, num_data = 0.0, 0
            for graph_data, smiles_list in tqdm(test_loader, desc="Testing"):
                graph_data = graph_data.to(self.device)
                __, pred = model(graph_data, smiles_list)
                loss = self._step(model, graph_data, smiles_list)
                test_loss += loss.item() * graph_data.y.size(0)
                num_data += graph_data.y.size(0)

                if self.normalizer: pred = self.normalizer.denorm(pred)
                if self.config['dataset']['task'] == 'classification': pred = F.softmax(pred, dim=-1)

                predictions.extend(pred.cpu().detach().numpy())
                labels.extend(graph_data.y.cpu().flatten().numpy())
            test_loss /= num_data
        
        if self.config['dataset']['task'] == 'regression':
            predictions, labels = np.array(predictions), np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print(f'Test Loss: {test_loss:.4f}, Test MAE: {mae:.4f}')
                return mae  # 최종 점수 반환
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print(f'Test Loss: {test_loss:.4f}, Test RMSE: {rmse:.4f}')
                return rmse  # 최종 점수 반환
        elif self.config['dataset']['task'] == 'classification':
            predictions, labels = np.array(predictions), np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print(f'Test Loss: {test_loss:.4f}, Test ROC AUC: {roc_auc:.4f}')
            return roc_auc  


def main(config):
    dataset = HybridDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    
    final_score = fine_tune.train()
    return final_score

if __name__ == "__main__":
    config = yaml.load(open("DRGS-Net.yaml", "r"), Loader=yaml.FullLoader)
    
    # config 파일에서 task_name을 직접 읽어옵니다.
    task_name = config['task_name']

    if task_name == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif task_name == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif task_name == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif task_name == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif task_name == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif task_name == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif task_name == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    # --- 아래는 regression task ---
    elif task_name == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif task_name == "ESOL":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif task_name == "Lipo":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif task_name == "qm7":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif task_name == "qm8":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif task_name == "qm9":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError(f'Undefined downstream task in config: {task_name}')

    # ===== 시행 횟수를 포함한 결과 폴더 경로 생성 =====
    base_results_dir = 'experiments'
    run_number = 1
    while True:
        results_dir = os.path.join(base_results_dir, f"{task_name}_{run_number}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            break
        run_number += 1
    
    print(f"\n\n{'='*20} Starting Experiment for: {task_name} (Run #{run_number}) {'='*20}")
    print(f"Results for this run will be saved in: {results_dir}")
    print("Running on targets:", target_list)

    results_list = []
    # 현재 데이터셋의 모든 타겟을 순회하며 학습
    for target in target_list:
        print(f"\n===== Training for target: {target} =====")
        config['dataset']['target'] = target
        
        result_score = main(config)
        results_list.append([target, result_score])

    # ===== 한 데이터셋의 모든 타겟에 대한 실험이 끝나면 CSV로 저장 =====
    
    # 1. 결과를 데이터프레임으로 변환 (이 부분에서 타겟별로 기록됩니다)
    metric_name = 'ROC_AUC' if config['dataset']['task'] == 'classification' else 'MAE/RMSE'
    df = pd.DataFrame(results_list, columns=['Target', metric_name])
    
    # 2. 평균/표준편차를 계산 (타겟이 2개 이상일 때만 실행됩니다)
    if len(target_list) > 1:
        mean_score = df[metric_name].mean()
        std_score = df[metric_name].std()
        
        summary_df = pd.DataFrame([
            {'Target': '---', metric_name: '---'},
            {'Target': 'Mean', metric_name: mean_score},
            {'Target': 'Std', metric_name: std_score}
        ])
        df = pd.concat([df, summary_df], ignore_index=True)
    
    # 3. CSV 파일로 저장
    results_csv_path = os.path.join(results_dir, 'results.csv')
    df.to_csv(results_csv_path, index=False)
    
    print(f"\n{'='*20} Experiment for {task_name} (Run #{run_number}) Finished {'='*20}")
    print(f"Results saved to: {results_csv_path}")
    print(df) # 최종 데이터프레임 출력
    print("\n\n")
