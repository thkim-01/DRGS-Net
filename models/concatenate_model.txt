import torch
from torch import nn

# Roberta 클래스를 직접 임포트합니다.
from transformers import RobertaModel, RobertaTokenizer

# 기존 MolCLR 모델 구조를 가져옵니다.
from models.ginet_finetune import GINet

class HybridModel(nn.Module):
    def __init__(self, molclr_config, task_type, chemberta_model_name="seyonec/ChemBERTa-77M-MLM"):
        super(HybridModel, self).__init__()

        self.molclr_model = GINet(task_type, **molclr_config)
        molclr_embedding_dim = self.molclr_model.pred_head[0].in_features

        self.chemberta_model = RobertaModel.from_pretrained(chemberta_model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(chemberta_model_name)
        
        chemberta_embedding_dim = self.chemberta_model.config.hidden_size

        combined_embedding_dim = molclr_embedding_dim + chemberta_embedding_dim
        output_dim = 2 if task_type == 'classification' else 1
        self.hybrid_pred_head = nn.Sequential(
            nn.Linear(combined_embedding_dim, combined_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combined_embedding_dim // 2, output_dim)
        )
    def forward(self, graph_data, smiles_list):
        # 1. MolCLR 경로
        graph_embedding, _ = self.molclr_model(graph_data)

        # 2. ChemBERTa 경로 (수동 패딩 및 절단)
        all_input_ids = []
        all_attention_masks = []
        max_length = 128
        
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        for smile in smiles_list:
            token_ids = self.tokenizer.encode(smile, add_special_tokens=False)
            if len(token_ids) > max_length - 2:
                token_ids = token_ids[:max_length - 2]
            
            input_ids = [cls_token_id] + token_ids + [sep_token_id]
            attention_mask = [1] * len(input_ids)
            
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long)

        chemberta_inputs = {
            'input_ids': input_ids_tensor.to(graph_embedding.device),
            'attention_mask': attention_mask_tensor.to(graph_embedding.device)
        }
        
        chemberta_outputs = self.chemberta_model(**chemberta_inputs)

        last_hidden_state = chemberta_outputs[0]
        smiles_embedding = last_hidden_state[:, 0]

        combined_embedding = torch.cat([graph_embedding, smiles_embedding], dim=1)
        prediction = self.hybrid_pred_head(combined_embedding)
        
        return combined_embedding, prediction

    def load_molclr_pre_trained_weights(self, state_dict):
        self.molclr_model.load_my_state_dict(state_dict)
        print("Loaded pre-trained MolCLR weights with success.")