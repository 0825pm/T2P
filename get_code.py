import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
import einops
from common.arguments import parse_args
from common.utils import Load_model
from dataset.data_condition import load_data, make_data_iter
from dataset.batch import Batch
from model.vqae_merge import QAE
from transformers import AutoTokenizer, GemmaForCausalLM
import logging
import torch.nn.functional as F

def _save_as_text_format(data_array, file_path):
    """
    NumPy 배열을 텍스트 포맷으로 저장합니다.
    (B, 1, D) -> (B, D)로 평탄화하여 저장합니다.
    """
    with open(file_path, 'w') as f:
        for i in range(data_array.shape[0]):
            flattened_data = data_array[i].reshape(-1)
            line = ' '.join(map(str, flattened_data))
            f.write(line + '\n')

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.safe_load(ymlfile)

def get_text_context_vector(tokenizer, text_model, text_input, device):
        """Text input을 Gemma에 통과시켜 단일 Context Vector (B, 1, H)를 추출합니다."""
        
        
        # 1. Text Encoding
        tokenizer.padding_side = 'left' 
        tokenized = tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Gemma CausalLM forward pass (Hidden State 반환)
        gemma_output = text_model(
            **tokenized, 
            output_hidden_states=True,
            return_dict=True
        )
        
        # 마지막 레이어의 Hidden State 추출 (B, L_text, H_gemma)
        last_hidden_state = gemma_output.hidden_states[-1] 
        attention_mask = tokenized['attention_mask'] # (B, L_text)

        # 2. Context Vector 생성 (Non-padding 토큰에 대한 평균 풀링)
        text_emb = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-5)
                
        return text_emb

def extract_latents(args, config, model, phase="train"):
    
    logging.info(f"Starting latent & context extraction for {phase} set...")
    gemma_model_name = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
    # Gemma는 Multimodal AR Backbone으로 사용됩니다.
    text_model = GemmaForCausalLM.from_pretrained(gemma_model_name).to(device)
    # 1. Load data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    
    if phase == "train":
        data = train_data
    elif phase == "dev":
        data = dev_data
    elif phase == "test":
        data = test_data
    else:
        raise ValueError("Invalid phase")
        
    dataloader = make_data_iter(data,
                                batch_size=args.batch_size, 
                                batch_type="sentence",
                                train=False, shuffle=False)
    
    all_latents = []
    all_contexts = []  # conditions 대신 contexts로 명칭 변경 (의미 명확화)
    all_texts = []
    
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader, desc=f"Extracting {phase}")):
            
            batch = Batch(torch_batch=batch_data,
                          pad_index=0,
                          model=model) 

            # 1. Text Input 준비
            text_input = batch.text
            
            # 2. Pose Input 준비
            pose_input = batch.trg_input[:, :, :150]
            pose_input = einops.rearrange(pose_input, "b f (n c) -> b f n c", c=3)
            pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
            
            # 3. Pose Latent (z) 추출 (QAE)
            encoded_feat = model.encode_pose(pose_input, pose_length) # [B, T, H]
        
            # 2. Bottleneck (Q-Former + VQ if needed)
            qae_feat, indices = model.qformer(encoded_feat)
            current_len = indices.shape[1]
            pad_amount = 300 - current_len

            if pad_amount > 0:
                # F.pad(input, (left, right), value=...)
                indices_padded = F.pad(indices, (0, pad_amount), value=-1)
            else:
                # 만약 300보다 길다면 자름 (선택 사항)
                indices_padded = indices[:, :300]
            # z_gt_seq: (B, 24, 64)
            # z_gt_seq = einops.rearrange(qae_feat, 'b h t j -> b (t j) h') 
            
            # 4. Context Vector 추출 (핵심 변경 사항)
            # Gemma를 통해 텍스트를 인코딩하고 Projection한 결과인 Context Vector만 추출합니다.
            # Diffusion 학습 시에는 이 Vector를 로드하여 바로 Motion Latent와 결합합니다.
            
            # (B, 1, gemma_hidden_size)
            context_vector = get_text_context_vector(tokenizer, text_model, text_input, batch.src.device)
            
            # Store numpy arrays
            all_latents.append(indices_padded.cpu().numpy())
            all_contexts.append(context_vector.cpu().numpy()) 
            all_texts.extend(text_input)
            
    # Concatenate and save
    final_latents_array = np.concatenate(all_latents, axis=0)
    final_contexts = np.concatenate(all_contexts, axis=0)
    final_texts = np.array(all_texts)
    
    latent_save_dir = os.path.join(os.path.dirname(args.checkpoint.rstrip('/')), "latents")
    os.makedirs(latent_save_dir, exist_ok=True)
    
    # 1. 시퀀스 잠재 벡터 저장 (.latent)
    _save_as_text_format(final_latents_array, 
                         os.path.join(latent_save_dir, f"{phase}.code"))

    # 2. Context 벡터 저장 (.cond)
    # 파일 확장자는 기존 코드와의 호환성을 위해 .cond를 유지하지만, 내용은 Context Vector입니다.
    _save_as_text_format(final_contexts, 
                         os.path.join(latent_save_dir, f"{phase}.cond"))
    
    # 3. 텍스트 원본 저장 (.ntext)
    _save_as_text_format(final_texts, 
                         os.path.join(latent_save_dir, f"{phase}.ntext"))
    
    logging.info(f"Saved sequence latents to {phase}.latent, Context vectors to {phase}.cond")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")

    if args.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'get_latent.log'), level=logging.INFO)

    print(f"Loading model: {args.model}")
    
    # 모델 초기화
    model = QAE(config["model"]["qae"]).to(device)
    
    # QAE 가중치 로드 (Phase 1에서 학습된 것)
    print("Loading QAE weights...")
    class DummyArgs: pass
    dummy_args = DummyArgs()
    dummy_args.previous_dir = args.previous_dir 
    Load_model(dummy_args, model) 
    
    # 3. Extract and Save
    print("EXTRACTING TRAIN DATA LATENT & CONTEXT")
    extract_latents(args, config, model, "train")
    
    print("EXTRACTING DEV DATA LATENT & CONTEXT")
    extract_latents(args, config, model, "dev")
    
    print("EXTRACTING TEST DATA LATENT & CONTEXT")
    extract_latents(args, config, model, "test")