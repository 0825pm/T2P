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
from model.orthus import GEMMA 
import logging

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

def extract_latents(args, config, model, phase="train"):
    
    logging.info(f"Starting latent & context extraction for {phase} set...")
    
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
            body_feat, rhand_feat, lhand_feat = model.qae.encode_pose(pose_input, pose_length)
            qae_feat = model.qae.qformer(body_feat, rhand_feat, lhand_feat)
            
            # z_gt_seq: (B, 24, 64)
            z_gt_seq = einops.rearrange(qae_feat, 'b h t j -> b (t j) h') 
            
            # 4. Context Vector 추출 (핵심 변경 사항)
            # Gemma를 통해 텍스트를 인코딩하고 Projection한 결과인 Context Vector만 추출합니다.
            # Diffusion 학습 시에는 이 Vector를 로드하여 바로 Motion Latent와 결합합니다.
            
            # (B, 1, gemma_hidden_size)
            context_vector = model._get_text_context_vector(text_input, batch.src.device)
            
            # Store numpy arrays
            all_latents.append(z_gt_seq.cpu().numpy())
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
                         os.path.join(latent_save_dir, f"{phase}.latent"))

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
    model = GEMMA(config["model"]).to(device)
    
    # QAE 가중치 로드 (Phase 1에서 학습된 것)
    print("Loading QAE weights...")
    class DummyArgs: pass
    dummy_args = DummyArgs()
    dummy_args.previous_dir = args.previous_dir 
    Load_model(dummy_args, model.qae) 
    
    # 3. Extract and Save
    print("EXTRACTING TRAIN DATA LATENT & CONTEXT")
    extract_latents(args, config, model, "train")
    
    print("EXTRACTING DEV DATA LATENT & CONTEXT")
    extract_latents(args, config, model, "dev")
    
    print("EXTRACTING TEST DATA LATENT & CONTEXT")
    extract_latents(args, config, model, "test")