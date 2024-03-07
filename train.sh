python scripts/run_flow.py --data_dir waymo_dynamic/waymo/processed/training/016
python scripts/run_flow.py --data_dir waymo_dynamic/waymo/processed/training/021
python scripts/run_flow.py --data_dir waymo_dynamic/waymo/processed/training/022

python DPT/run_monodepth.py --input_path waymo_dynamic/waymo/processed/training/016/images --output_path waymo_dynamic/waymo/processed/training/016/depth --model_type dpt_large

CUDA_VISIBLE_DEVICES=1 python localTensoRF/train.py --downsampling 2 --datadir waymo_dynamic/waymo/processed/training/016 --logdir out --fov 69 --N_voxel_init 32768000 --N_voxel_final 32768000 --vis_every 2000
CUDA_VISIBLE_DEVICES=2 python localTensoRF/train.py --downsampling 2 --datadir waymo_dynamic/waymo/processed/training/016 --logdir out_normal --fov 69 --N_voxel_init 32768000 --N_voxel_final 32768000 --vis_every 2000


CUDA_VISIBLE_DEVICES=3 python localTensoRF/train.py --downsampling 2 --datadir waymo_dynamic/waymo/processed/training/016 --logdir log/016_change --fov 69 --N_voxel_init 16384000 --N_voxel_final 16384000 --vis_every 1000 --use_dynamic 1
