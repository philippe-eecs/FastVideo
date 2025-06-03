import csv
import json
import os
import argparse
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.io
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel
# Assuming diffusers.AutoencoderKL might be used directly or indirectly by VAELoader
# If VAELoader loads a custom VAE, AutoencoderKL import might not be directly needed here
# but often part of the ecosystem.
from diffusers import AutoencoderKL

from fastvideo.v1.logger import init_logger
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.models.loader.component_loader import VAELoader
# Attempt to import the schema directly
from fastvideo.v1.dataset.dataloader.schema import pyarrow_schema

logger = init_logger(__name__)

def setup_distributed(rank, world_size):
    """Initialize distributed processing."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed processing."""
    dist.destroy_process_group()

def get_video_metadata(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames:format=duration',
            '-of', 'json', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if not data.get('streams') or not data.get('format'):
            raise ValueError(f"No stream/format in ffprobe output for {video_path}")
        
        stream = data['streams'][0]
        format_info = data['format']
        
        # Parse frame rate
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        return {
            "fps": fps,
            "duration_sec": float(format_info.get('duration', 0.0)),
            "width": int(stream['width']),
            "height": int(stream['height']),
            "num_frames": int(stream.get('nb_frames', 0))
        }
    except Exception as e:
        logger.warning(f"Metadata extraction failed for {video_path}: {e}")
        # Return default values
        return {
            "fps": 30.0,
            "duration_sec": 0.0,
            "width": 640,
            "height": 480,
            "num_frames": 0
        }

def process_video(video_path: str, vae, device, target_frames: int = 81, frame_sample_rate: int = 4):
    """Load and process a single video through VAE."""
    try:
        # Load video
        video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")
        if video_frames.shape[0] == 0:
            return None
        
        # Sample frames
        if frame_sample_rate > 1:
            indices = torch.arange(0, video_frames.shape[0], frame_sample_rate).long()
            video_frames = video_frames[indices]
        
        # Pad or sample to target frames
        current_frames = video_frames.shape[0]
        if current_frames > target_frames:
            indices = torch.linspace(0, current_frames - 1, target_frames).long()
            video_frames = video_frames[indices]
        elif current_frames < target_frames:
            pad_size = target_frames - current_frames
            last_frame = video_frames[-1:].repeat(pad_size, 1, 1, 1)
            video_frames = torch.cat([video_frames, last_frame], dim=0)
        
        # Normalize
        video_frames = video_frames.float() / 127.5 - 1.0
        
        # Process through VAE
        # video_frames is (T, C, H, W), need (B=1, C, T, H, W) for VAE
        video_tensor = video_frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            latents = vae.encode(video_tensor).sample()
        
        # latents is (1, LatentC, T_lat, H_lat, W_lat)
        return latents[0].cpu()
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return None

def process_text(caption: str, tokenizer, text_encoder, device):
    """Process text through tokenizer and encoder."""
    try:
        inputs = tokenizer(
            caption,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        
        return embeddings[0].cpu(), attention_mask[0].cpu()
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return None, None

def write_batch_to_parquet(batch_data: list, output_dir: str, batch_num: int, rank: int = 0):
    """Write batch to parquet with rank suffix to avoid conflicts."""
    if not batch_data:
        return
    try:
        table = pa.Table.from_pylist(batch_data, schema=pyarrow_schema)
        output_path = os.path.join(output_dir, f"data_chunk_rank{rank:02d}_{batch_num:04d}.parquet")
        pq.write_table(table, output_path, compression='zstd')
        logger.info(f"Rank {rank} wrote batch {batch_num} to {output_path}")
    except Exception as e:
        logger.error(f"Rank {rank} parquet writing error for batch {batch_num}: {e}")

def distributed_worker(rank: int, world_size: int, args):
    """Worker function for distributed processing."""
    setup_distributed(rank, world_size)
    device = f"cuda:{rank}"
    logger.info(f"Worker {rank}/{world_size} starting on {device}")
    
    # Load models
    try:
        # Load VAE
        vae_model_path = os.path.join(args.model_path, "vae")
        vae_config = WanVAEConfig(load_encoder=True, load_decoder=False)
        fv_args = FastVideoArgs(
            model_path=vae_model_path,
            vae_config=vae_config,
            vae_precision="fp32"
        )
        fv_args.device = device
        vae = VAELoader().load(
            model_path=vae_model_path,
            architecture="",
            fastvideo_args=fv_args
        )
        vae.eval()
        
        # Load tokenizer
        tokenizer_path = os.path.join(args.model_path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            model_max_length=512
        )
        
        # Load text encoder
        text_encoder_path = os.path.join(args.model_path, "text_encoder")
        text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path).to(device)
        text_encoder.eval()
        
        logger.info(f"Rank {rank}: All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Rank {rank}: Model loading failed: {e}")
        cleanup_distributed()
        return
    
    # Read CSV and count total rows
    with open(args.csv_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = list(csv.DictReader(csvfile))
        total_rows = len(reader)
    
    # Calculate this rank's portion
    rows_per_rank = total_rows // world_size
    start_idx = rank * rows_per_rank
    end_idx = start_idx + rows_per_rank if rank < world_size - 1 else total_rows
    
    logger.info(f"Rank {rank}: Assigned rows {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")
    
    # Pre-filter: Check which videos exist in this rank's range
    logger.info(f"Rank {rank}: Pre-filtering videos...")
    valid_entries = []
    missing_count = 0
    
    for idx in range(start_idx, end_idx):
        row = reader[idx]
        video_path_suffix = row.get(args.path_column)
        caption = row.get(args.caption_column)
        
        if not video_path_suffix or not caption:
            continue
        
        video_path = os.path.join(args.video_base_dir, video_path_suffix)
        
        if os.path.exists(video_path):
            valid_entries.append({
                'idx': idx,
                'video_path': video_path,
                'caption': caption,
                'basename': os.path.basename(video_path)
            })
        else:
            missing_count += 1
    
    logger.info(f"Rank {rank}: Found {len(valid_entries)} valid videos, {missing_count} missing videos")
    
    if not valid_entries:
        logger.warning(f"Rank {rank}: No valid videos found, exiting")
        cleanup_distributed()
        return
    
    # Process valid entries
    local_records = []
    batch_num = 0
    
    # Create output directory (all ranks can do this safely)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for entry in tqdm(valid_entries, desc=f"Rank {rank}", position=rank):
        video_path = entry['video_path']
        caption = entry['caption']
        idx = entry['idx']
        
        # Process video
        latents = process_video(
            video_path, vae, device,
            args.target_frames, args.frame_sample_rate
        )
        if latents is None:
            logger.warning(f"Rank {rank}: Failed to process video {video_path}")
            continue
        
        # Process text
        text_emb, text_mask = process_text(caption, tokenizer, text_encoder, device)
        if text_emb is None or text_mask is None:
            logger.warning(f"Rank {rank}: Failed to process text for {video_path}")
            continue
        
        # Get metadata
        metadata = get_video_metadata(video_path)
        
        # Create record
        record = {
            "id": os.path.splitext(entry['basename'])[0] + f"_{idx}",
            "file_name": entry['basename'],
            "caption": caption,
            "media_type": "video",
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "duration_sec": metadata["duration_sec"],
            "num_frames": metadata["num_frames"],
            "vae_latent_bytes": latents.numpy().tobytes(),
            "vae_latent_shape": list(latents.shape),
            "vae_latent_dtype": str(latents.dtype).replace("torch.", ""),
            "text_embedding_bytes": text_emb.numpy().tobytes(),
            "text_embedding_shape": list(text_emb.shape),
            "text_embedding_dtype": str(text_emb.dtype).replace("torch.", ""),
            "text_attention_mask_bytes": text_mask.numpy().astype(np.uint8).tobytes(),
            "text_attention_mask_shape": list(text_mask.shape),
            "text_attention_mask_dtype": "uint8",
        }
        
        local_records.append(record)
        
        # Write parquet file when batch size is reached
        if len(local_records) >= args.parquet_batch_size:
            write_batch_to_parquet(local_records, args.output_dir, batch_num, rank)
            local_records = []
            batch_num += 1
    
    # Write any remaining records
    if local_records:
        write_batch_to_parquet(local_records, args.output_dir, batch_num, rank)
    
    logger.info(f"Rank {rank}: Successfully processed {len(valid_entries)} videos")
    
    # Wait for all processes to complete before exiting
    dist.barrier()
    
    cleanup_distributed()

def main(args):
    """Main function that spawns distributed workers."""
    world_size = torch.cuda.device_count()
    if world_size < 1:
        logger.error("No CUDA devices available!")
        return
    
    if args.num_gpus > 0 and args.num_gpus <= world_size:
        world_size = args.num_gpus
    
    logger.info(f"Starting distributed processing with {world_size} GPUs")
    
    if world_size == 1:
        distributed_worker(0, 1, args)
    else:
        mp.spawn(distributed_worker, args=(world_size, args), nprocs=world_size, join=True)
    
    logger.info("All processing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple distributed video-text dataset conversion")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("video_base_dir", type=str, help="Base directory for videos")
    parser.add_argument("output_dir", type=str, help="Output directory for parquet files")
    parser.add_argument("--model_path", type=str, 
                       default="/fsx-project/philippehansen/models/Wan2.1-T2V-1.3B-Diffusers",
                       help="Path to model directory")
    parser.add_argument("--path_column", type=str, default="file_path", 
                       help="CSV column containing video paths")
    parser.add_argument("--caption_column", type=str, default="dense_caption",
                       help="CSV column containing captions")
    parser.add_argument("--parquet_batch_size", type=int, default=128,
                       help="Records per parquet file")
    parser.add_argument("--target_frames", type=int, default=81,
                       help="Target number of frames")
    parser.add_argument("--frame_sample_rate", type=int, default=4,
                       help="Frame sampling rate")
    parser.add_argument("--num_gpus", type=int, default=0,
                       help="Number of GPUs to use (0=all)")
    
    args = parser.parse_args()
    main(args) 