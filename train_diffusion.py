import os
import torch
import random
import numpy as np
import json
import einops
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# 사용자님의 프로젝트에서 필요한 모듈 (경로를 맞게 수정해주세요)
from common.arguments import parse_args
from model.llama_model import LLaMAHF, LLaMAHFConfig
from utils.helpers import load_config
from utils.utils_model import initial_optim, get_logger

# 1. 사용자 데이터셋 로드
class LatentDataset(Dataset):
    def __init__(self, latent_path, cond_path):
        # 파일 로드 (get_latent.py의 출력을 가정)
        self.latents = np.load(latent_path, allow_pickle=True)  # (Total_Sequences, Max_Len, D_latent)
        self.conditions = np.load(cond_path, allow_pickle=True)  # (Total_Sequences, D_text)
        assert len(self.latents) == len(self.conditions), "Latents and Conditions length mismatch."
        
        # MotionStreamer의 m_tokens_len을 직접 추출할 수 없으므로,
        # 편의상 최대 길이와 마스크를 가정하거나, 실제 데이터셋 구현이 필요함.
        # 여기서는 길이를 모두 최대로 가정하고, 실제 데이터 로더가 마스크를 제공한다고 가정.

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # m_tokens_len은 MotionStreamer에서 필요하지만, 여기서는 단순하게 최대 길이로 가정
        m_tokens_len = 24
        return self.conditions[idx], self.latents[idx], m_tokens_len

def collate_fn(batch):
    # DataLoader가 NumPy 배열을 Tensor로 변환하고 배치화
    feat_text, m_tokens, m_tokens_len = zip(*batch)
    
    # 텐서로 변환
    feat_text = torch.from_numpy(np.stack(feat_text)).float()
    m_tokens = torch.from_numpy(np.stack(m_tokens)).float()
    m_tokens_len = torch.tensor(m_tokens_len).long() # 실제로는 가변 길이의 길이를 받아야 합니다.
    
    m_tokens = einops.rearrange(m_tokens, 'b h t c -> b (t c) h')
    
    return feat_text, m_tokens, m_tokens_len

# 2. 스케줄러 정의
class WarmupCosineDecayScheduler:
    # train_t2m.py의 스케줄러 정의를 그대로 사용
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=min_lr)
        
    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter):
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            # CosineAnnealingLR은 현재 step을 기준으로 T_max 이내에서 스케줄링하므로,
            # warmup 이터를 뺀 값으로 계산합니다.
            self.cosine_scheduler.step(current_iter - self.warmup_iters) 
            
    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'total_iters': self.total_iters,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters = state_dict['total_iters']
        self.min_lr = state_dict['min_lr']

# 3. Two-Forward 전략을 위한 유틸리티 함수
def lengths_to_mask(lengths, max_len):
    # train_t2m.py의 함수를 그대로 사용
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    # MotionStreamer의 cosine_decay 로직 (노출 비율 γ_t 제어)
    step = torch.tensor(step, dtype=torch.float32)  
    total_steps = torch.tensor(total_steps, dtype=torch.float32)  
    
    # MotionStreamer 논문 (Sec.A)의 코사인 스케줄: γ_t = 0.5 * (1 - cos(πt/T))를 따릅니다.
    cosine_factor = 0.5 * (1 - torch.cos(torch.pi * step / total_steps))
    
    # cos_factor가 0(t=0)에서 1(t=T)로 변하므로, start_value와 end_value 사이를 이동합니다.
    return start_value + (end_value - start_value) * cosine_factor

def replace_with_pred(latents, pred_xstart, step, total_steps):
    # MotionStreamer의 Two-Forward Latent Replacement 로직
    
    # Two-Forward 전략에서 MotionStreamer는 γ_t = 0.5 * (1 - cos(πt/T))를 사용
    # t=0일 때, γ_t=0 (교체 없음), t=T일 때, γ_t=1 (전체 교체)
    decay_factor = cosine_decay(step, total_steps, start_value=0.0, end_value=1.0).to(latents.device)
    
    b, l, d = latents.shape
    
    # 교체할 토큰 개수: 전체 길이 * decay_factor
    # MotionStreamer 논문 (Sec.A)의 γ_t는 '교체된 모션 토큰의 비율'을 의미합니다.
    num_replace_float = l * decay_factor
    num_replace = int(num_replace_float.item()) # 스케일러 값을 item()으로 추출

    # 무작위로 교체 인덱스 선택
    replace_indices = torch.randperm(l)[:num_replace]  

    # 마스크 생성 (B, L)
    replace_mask = torch.zeros(b, l, dtype=torch.bool).to(latents.device)
    replace_mask[:, replace_indices] = 1  
    
    # (B, L, D)로 마스크 확장
    replace_mask_expanded = replace_mask.unsqueeze(-1).expand_as(latents)

    updated_latents = latents.clone()  
    
    # pred_xstart의 (B, L, D) 형태를 활용하여 교체
    updated_latents[replace_mask_expanded] = pred_xstart[replace_mask_expanded]
    
    return updated_latents

diffmlps_batch_mul = 4 # MotionStreamer에서 사용된 멀티플라이어
def forward_loss_withmask_2_forward(latents, trans, m_lens, feat_text, step, total_steps):
    """latents: gt motion latents, trans: AR Transformer model, feat_text: text condition"""
    
    # trans: trans_encoder (LLaMAHF)
    
    # --- First Forward: Prediction ---
    # AR Transformer를 통과시켜 Diffusion Head의 Condition (z)를 얻습니다.
    conditions = trans(latents, feat_text)  
    conditions = conditions.contiguous()
    z_for_pred = conditions[:,:-1,:] # MotionStreamer는 끝 토큰을 제외하고 사용
    
    b, l, d = latents.shape     
    
    # Diffusion Loss를 계산하고 pred_xstart를 얻기 위해 데이터를 평탄화합니다.
    target_for_pred = latents.clone().detach()       
    target_for_pred = target_for_pred.reshape(b * l, -1)    
    z_for_pred = z_for_pred.reshape(b * l, -1)            
    
    with torch.no_grad():
        # trans.diff_loss는 Diffusion Loss 계산 및 pred_xstart를 반환합니다.
        _, pred_xstart_flat = trans.diff_loss(target=target_for_pred, z=z_for_pred)  

    pred_xstart = pred_xstart_flat.clone().detach()
    pred_xstart = pred_xstart.reshape(b, l, -1)           

    # --- Second Forward: Loss Calculation ---
    # 1. Latent Replacement: GT의 일부를 1차 예측값으로 대체
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)    
    
    # 2. Second AR Forward: 업데이트된 Latent로 Condition 생성
    updated_conditions = trans(updated_latents, feat_text)  
    updated_conditions = updated_conditions.contiguous()
    updated_z = updated_conditions[:,:-1,:]      

    # 3. Loss 계산을 위한 데이터 준비 (마스크 및 배치 확장)
    mask = lengths_to_mask(m_lens, l)       
    mask = mask.reshape(b * l).repeat(diffmlps_batch_mul) # MotionStreamer의 배치 확장

    updated_target = latents.clone().detach()       

    updated_target = updated_target.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)    
    updated_z = updated_z.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)            

    # 마스크 적용
    updated_target = updated_target[mask]                   
    updated_z = updated_z[mask]                            

    # 최종 Diffusion Loss 계산
    updated_loss, _ = trans.diff_loss(target=updated_target, z=updated_z)  

    return updated_loss


# 4. Main Training Loop
if __name__ == '__main__':
    # 설정 로드
    args = parse_args()
    config = load_config(args.config)
    
    # 디렉토리 및 가속기 설정
    args.out_dir = os.path.join(args.checkpoint.rstrip('/'), 'diffusion_checkpoint')
    os.makedirs(args.out_dir, exist_ok = True)

    # 로거 설정
    logger = get_logger(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # 데이터 로더 설정 (경로를 실제 파일 위치로 수정해야 합니다)
    LATENT_DIR = os.path.join(os.path.dirname(args.previous_dir.rstrip('/')), "latents")
    try:
        latent_dataset = LatentDataset(
            latent_path=os.path.join(LATENT_DIR, "train_latents.npy"),
            cond_path=os.path.join(LATENT_DIR, "train_conditions.npy")
        )
        train_loader = DataLoader(
            latent_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
    except Exception as e:
        logger.info(f"🚨 Error loading latent data: {e}")
        logger.info("Please ensure 'train_latents.npy' and 'train_conditions.npy' exist and match the expected format.")
        exit()

    # 모델 설정
    # MotionStreamer 논문 참고: dim=768, layers=12, heads=12. 
    # QAE 설정에서 필요한 값을 가져와야 합니다.
    MODEL_CONFIG = config["model"]["qae"]
    TRAING_CONFIG = config["training"]
    
    # AR Transformer 모델의 hidden_size를 가져옵니다.
    AR_DIM = MODEL_CONFIG["hidden_size"] 
    # Diffusion Head의 레이어 수는 논문에서 9 layers (LLaMA-12 기준)입니다.
    NUM_DIFFUSION_HEAD_LAYERS = 9 
    
    # LLaMA-like AR Transformer 초기화 (LLaMAHFConfig를 사용한다고 가정)
    llama_config = LLaMAHFConfig.from_name('Normal_size')
    # 사용자 QAE 설정의 hidden_size를 사용
    llama_config.latent_dim = AR_DIM 
    
    # LLaMAHF 모델 초기화 (실제로는 llama_model.py에서 로드해야 합니다)
    try:
        # 가상의 LLaMAHF 로드
        trans_encoder = LLaMAHF(llama_config, NUM_DIFFUSION_HEAD_LAYERS, AR_DIM, "cuda") 
        trans_encoder.train()
        trans_encoder.to("cuda")
    except Exception as e:
        logger.info(f"🚨 Error initializing LLaMAHF model: {e}")
        logger.info("Please ensure LLaMAHF and LLaMAHFConfig classes are correctly defined and imported.")
        exit()


    # 옵티마이저 및 스케줄러
    optimizer = initial_optim(
        'all', TRAING_CONFIG["learning_rate"], 1e-6, trans_encoder, TRAING_CONFIG["optimizer"]
    )
    
    # total_iter 설정 (예: 에포크 * 배치 수)
    TOTAL_ITERS = TRAING_CONFIG["epochs"] * len(train_loader)
    
    scheduler = WarmupCosineDecayScheduler(
        optimizer, TOTAL_ITERS // 10, TOTAL_ITERS, min_lr=TRAING_CONFIG["min_lr"]
    )
    
    train_loader_iter = iter(train_loader)


    # 학습 루프
    nb_iter, avg_loss = 0, 0.

    while nb_iter <= TOTAL_ITERS:
        try:
            feat_text, m_tokens, m_tokens_len = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            feat_text, m_tokens, m_tokens_len = next(train_loader_iter)
            
        feat_text, m_tokens = feat_text.to("cuda"), m_tokens.to("cuda")

        # --------- GT Latents 준비 -------- 
        # MotionStreamer의 m_tokens는 End Token을 포함하며, input_latent는 End Token을 제외합니다.
        # 여기서는 편의상 m_tokens의 마지막을 End Token으로 가정하고 제외합니다.
        input_latent = m_tokens # 연속적인 토큰 (End Token 제외)
        
        # --------- Two-Forward Loss 계산 ---------

        loss = forward_loss_withmask_2_forward(
            latents=input_latent, 
            trans=trans_encoder, 
            m_lens=m_tokens_len, 
            feat_text=feat_text, 
            step=nb_iter, 
            total_steps=TOTAL_ITERS
        )

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(nb_iter)

        avg_loss = avg_loss + loss.item()

        nb_iter += 1
        
        # --------- 로깅 및 저장 ---------
        args.print_iter = args.logging_freq if hasattr(args, 'logging_freq') else 100
        if nb_iter % args.print_iter ==  0 :
            
            avg_loss = avg_loss / args.print_iter
            # writer.add_scalar('./Loss/train', avg_loss, nb_iter) # TensorBoard 로깅 필요
            current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('./LR/train', current_lr, nb_iter) # TensorBoard 로깅 필요
            msg = f"Train. Iter {nb_iter}/{TOTAL_ITERS} : Loss. {avg_loss:.5f} | LR. {current_lr:.6f}"
            logger.info(msg)
            avg_loss = 0.


        args.save_iter = args.validation_freq if hasattr(args, 'validation_freq') else 10000
        if nb_iter % args.save_iter == 0:
            
            torch.save({
                'trans': trans_encoder.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.out_dir, f'latest_iter_{nb_iter}.pth'))