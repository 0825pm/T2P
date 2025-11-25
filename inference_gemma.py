import os
import torch
import numpy as np
import yaml
import argparse
import logging
from tqdm import tqdm
import einops
import time

# 기존 프로젝트 모듈 임포트
from common.arguments import parse_args
from dataset.data_orthus import load_data, make_data_iter
from dataset.batch import Batch
from model.gemma_no_diffusion import GEMMA
from utils.plot_videos import plot_video, alter_DTW_timing
from common.utils import calculate_dtw

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.safe_load(ymlfile)
    
def create_mask(seq_lengths, device="cpu"):
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.bool()

def evaluate(args, config, dataloader, model, src_vocab):
    model.eval()
    
    # 결과 저장 경로 설정
    results_dir = os.path.join(args.checkpoint, "inference_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        format='%(asctime)s %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S', 
        filename=os.path.join(results_dir, 'inference.log'), 
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    all_dtw = []
    all_mpjpe = []
    
    logging.info(f"Starting inference on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            
            # 1. Batch 데이터 준비 (train_gemma.py와 동일)
            batch = Batch(torch_batch=batch_data,
                          pad_index=0,
                          model=model)
            
            # Text Input (List of strings)
            text_input = batch.text
            
            # Ground Truth Pose (B, T, 150) -> (B, T, 50, 3)
            # batch.trg_input은 (B, T, 151)이며 마지막 1차원은 Counter입니다.
            gt_pose_flat = batch.trg_input[:, :, :150]
            gt_pose = einops.rearrange(gt_pose_flat, "b f (n c) -> b f n c", c=3)
            
            # GT Lengths
            pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
            
            # 2. Pose Generation (NAR 방식)
            # (B, T, 50, 3)
            generated_pose = model.generate(text_input, diffusion_steps=args.steps, target_length=pose_length)
            
            # 3. 후처리 및 평가 준비
            # Masking (길이에 맞게)
            pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
            generated_pose = generated_pose.to(torch.float32) * pose_mask
            
            # 4. Metric 계산
            # MPJPE (Mean Per Joint Position Error)
            # (B,)
            joint_error = torch.mean(torch.norm(generated_pose - gt_pose, dim=-1), dim=[-1, -2])
            all_mpjpe.extend(joint_error.cpu().numpy().tolist())
            
            # DTW (Dynamic Time Warping)
            dtw_scores = calculate_dtw(gt_pose, generated_pose.cpu().detach())
            all_dtw.extend(dtw_scores)
            
            # 5. 시각화 (Video Save) - 첫 번째 배치거나 일부 샘플만 저장
            if batch_idx < 5: # 처음 5개 배치에 대해서만 영상 저장
                num_samples = min(1, generated_pose.shape[0]) # 배치 당 2개씩
                
                for i in range(num_samples):
                    gt_len_i = pose_length[i].item()
                    
                    # (T, 150) 형태로 변환
                    gt_pose_np_i = gt_pose[i, :gt_len_i].reshape(-1, 150)
                    pred_pose_np_i = generated_pose[i, :gt_len_i].reshape(-1, 150)
                    
                    # DTW 시각화를 위해 Counter(마지막 차원) 복구 (GT의 Counter 사용)
                    # train_gemma.py의 save_videos 로직 참조
                    counter = batch.trg_input[i, :gt_len_i, 150:]
                    gt_with_counter = torch.cat((gt_pose_np_i, counter), dim=-1)
                    pred_with_counter = torch.cat((pred_pose_np_i, counter), dim=-1)
                    
                    # DTW Timing 맞추기
                    timing_hyp_seq, ref_seq_count, _ = alter_DTW_timing(pred_with_counter, gt_with_counter)
                    
                    text_str = text_input[i]
                    # 파일명 안전하게 변환
                    filename_str = text_str.replace(" ", "_").replace("/", "-")[:50]
                    video_name = f"batch{batch_idx}_{i:02d}_{filename_str}.mp4"
                    
                    try:
                        plot_video(
                            joints=timing_hyp_seq,
                            file_path=results_dir,
                            video_name=video_name,
                            references=ref_seq_count, # GT를 Reference로 함께 출력
                            skip_frames=1,
                            sequence_ID=text_str
                        )
                    except Exception as e:
                        logging.error(f"Failed to plot video {video_name}: {e}")

    # 최종 결과 출력
    avg_mpjpe = np.mean(all_mpjpe) * 1000 # mm 단위 변환
    avg_dtw = np.mean(all_dtw)
    
    logging.info("="*30)
    logging.info(f"Inference Completed.")
    logging.info(f"Average MPJPE: {avg_mpjpe:.4f} mm")
    logging.info(f"Average DTW: {avg_dtw:.4f}")
    logging.info("="*30)

    print(f"MPJPE: {avg_mpjpe:.4f}, DTW: {avg_dtw:.4f}")

if __name__ == "__main__":
    # Argument Parsing (기존 common.arguments 활용하되 필요한 부분 추가)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/t2p_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to directory or specific .pth file')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--steps', type=int, default=50, help='Diffusion sampling steps')
    
    # train_gemma.py와 호환성을 위해 일부 더미 인자 추가 가능
    # 여기서는 필요한 것만 정의
    args = parser.parse_args()
    
    # Config 로드
    config = load_config(args.config)
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드 (Test Set)
    # train_gemma.py와 동일한 load_data 함수 사용
    print("Loading data...")
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    
    # Test Dataloader 생성
    batch_size = config["training"]["batch_size"]
    test_dataloader = make_data_iter(train_data,
                                     batch_size=batch_size,
                                     batch_type="sentence",
                                     train=False, 
                                     shuffle=False)
    
    # 2. 모델 초기화
    print("Loading model...")
    model = GEMMA(config["model"]).to(device)
    
    # 3. 체크포인트 로드
    if os.path.isdir(args.checkpoint):
        # 디렉토리인 경우 가장 최신 혹은 best 로드 (utils의 Load_model 활용 가능하지만 여기서는 직접 로드)
        # train_gemma.py는 Load_model(args, [model], ["model"]) 형식을 썼으나
        # 여기서는 단일 파일 지정을 권장하거나, 직접 로직 구현
        import glob
        pth_files = sorted(glob.glob(os.path.join(args.checkpoint, "*.pth")))
        if pth_files:
            ckpt_path = pth_files[-1] # 마지막 파일 (보통 last.pth나 best)
        else:
            raise FileNotFoundError(f"No .pth files found in {args.checkpoint}")
    else:
        ckpt_path = args.checkpoint
        
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 모델 가중치 로드 (DataParallel 등으로 저장되었을 경우 키 처리 필요할 수 있음)
    # 보통 model.state_dict()를 통째로 저장하므로 그대로 로드
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    # 4. 평가 실행
    evaluate(args, config, test_dataloader, model, src_vocab)