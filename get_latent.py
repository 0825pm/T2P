import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
import einops
from common.arguments import parse_args
from common.utils import Load_model
from dataset.data_qae import load_data, make_data_iter
from dataset.batch import Batch
from model.qae import QAE 
from model.text_encoder import T5XXLTextEncoder
import logging

def _save_as_text_format(data_array, file_path):
    """
    3D (B, L, D) 또는 2D (B, D) NumPy 배열을 각 샘플당 하나의 텍스트 라인으로 저장합니다.
    숫자는 공백으로 구분되며, 이는 dataset.data에서 .split(" ")으로 읽기 위함입니다.
    """
    with open(file_path, 'w') as f:
        for i in range(data_array.shape[0]):
            # 다차원 배열을 1차원으로 평탄화하여 하나의 긴 시퀀스로 만듭니다.
            # latent_seq: (L, D) -> (L*D)
            # condition: (D) -> (D)
            flattened_data = data_array[i].reshape(-1)
            # 숫자를 문자열로 변환하고 공백으로 연결합니다.
            # 소수점 정확도는 str()이 아닌, numpy.array2string 등을 사용해 제어할 수 있습니다.
            line = ' '.join(map(str, flattened_data))
            f.write(line + '\n')

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.safe_load(ymlfile)

def extract_latents(args, config, QAE_model, text_encoder, phase="train"):
    
    logging.info("Starting latent extraction...")
    
    # 1. Load data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    
    if phase == "train":
        data = train_data
    elif phase == "dev":
        data = dev_data
    elif phase == "test":
        data = test_data
    else:
        raise ValueError()
        
    
    # Latent for Training Data
    dataloader = make_data_iter(data,
                                    batch_size=args.batch_size, 
                                    batch_type="sentence",
                                    train=False, shuffle=False)
    
    all_latents = []
    all_conditions = []
    all_texts = []
    
    QAE_model.eval()
    text_encoder.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader, desc="Extracting Latents")):
            
            # Assuming QAE_model.use_cuda is True, Batch will move to CUDA
            batch = Batch(torch_batch=batch_data,
                          pad_index=0,
                          model=QAE_model) 

            # 1. Get Text Condition (c)
            # Convert token indices back to strings (excluding EOS/PAD)
            text_input = [" ".join([src_vocab.itos[batch.src[i][j]] 
                                    for j in range(len(batch.src[i])-1) 
                                    if src_vocab.itos[batch.src[i][j]] not in ['<pad>', '</s>']]) 
                                    for i in range(len(batch.src))]
            
            # The T5XXLEncoder needs to be on the same device as the batch
            condition_c = text_encoder(text_input) # (B, D_embed)
            
            # 2. Get Pose Latent (z)
            pose_input = batch.trg_input[:, :, :150]
            pose_input = einops.rearrange(pose_input, "b f (n c) -> b f n c", c=3)
            pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
            
            # QAE.encode outputs (B, 8, D_embed)
            body_feat, rhand_feat, lhand_feat = QAE_model.encode_pose(pose_input, pose_length)
        
            # Clustering
            latent_z = QAE_model.qformer(body_feat, rhand_feat, lhand_feat) # B, hidden, token, 3
            
            # Store numpy arrays
            all_latents.append(latent_z.cpu().numpy())
            all_conditions.append(condition_c.cpu().numpy())
            all_texts.extend(text_input)
            
    # Concatenate and save
    final_latents_array = np.concatenate(all_latents, axis=0)
    final_conditions = np.concatenate(all_conditions, axis=0)
    final_texts = np.array(all_texts)
    
    latent_save_dir = os.path.join(os.path.dirname(args.checkpoint.rstrip('/')), "latents")
    os.makedirs(latent_save_dir, exist_ok=True)
    
    # 1. 시퀀스 잠재 벡터 저장 (.latent 확장자)
    _save_as_text_format(final_latents_array, 
                         os.path.join(latent_save_dir, f"{phase}.latent"))

    # 2. 텍스트 조건 벡터 저장 (.cond 확장자)
    _save_as_text_format(final_conditions, 
                         os.path.join(latent_save_dir, f"{phase}.cond"))
    
    _save_as_text_format(final_texts, 
                         os.path.join(latent_save_dir, f"{phase}.ntext"))
    
    logging.info(f"Saved sequence latents to {phase}_latents.latent, conditions to {phase}_conditions.cond")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Setup GPU/Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")

    # Logging setup (re-use the logic from training.py)
    if args.train:
        logtime = os.path.basename(args.checkpoint).split('_')[-1]
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'get_latent.log'), level=logging.INFO)

    # 1. Instantiate and Load QAE Model
    # Assumes args.model = 'qae'
    QAE_module = QAE # Direct reference assuming it's imported correctly via exec
    model_config = config["model"]["qae"]
    QAE_model = QAE_module(model_config).to(device)
    
    # Load pre-trained QAE weights (assuming args.previous_dir points to Phase 1 checkpoint)
    # The Load_model expects a list of models and names
    class DummyArgs: pass
    dummy_args = DummyArgs()
    dummy_args.previous_dir = args.previous_dir
    Load_model(dummy_args, QAE_model) 
    
    # 2. Instantiate T5 Text Encoder
    text_encoder = T5XXLTextEncoder(embed_dim=model_config['hidden_size']).to(device) 

    # 3. Extract and Save
    print("EXTRACTING TRAIN DATA LATENT")
    extract_latents(args, config, QAE_model, text_encoder, "train")
    print("EXTRACTING DEV DATA LATENT")
    extract_latents(args, config, QAE_model, text_encoder, "dev")
    print("EXTRACTING TEST DATA LATENT")
    extract_latents(args, config, QAE_model, text_encoder, "test")