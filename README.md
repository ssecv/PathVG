(MICCAI 2025) PathVG: A New Benchmark and Dataset for Pathology Visual Grounding
This repository hosts the code, dataset, and pre-trained weights of the research presented in "PathVG: A New Benchmark and Dataset for Pathology Visual Grounding" (accepted by MICCAI 2025). It provides a comprehensive benchmark for pathology visual grounding tasks, along with reproducible experiments and model implementations.

## 📅 Updates (Timeline & To-Do)
Track key project milestones and resource releases below. Click the links to access corresponding resources once available:  
- [x] **2025-06-25**: Repository initialization (basic structure & README released)  
- [x] **2025-05-30**: PathVG Dataset public release → [Download Dataset](https://huggingface.co/datasets/fengluo/RefPath)   
- [ ] **2025-06-15**: Full training/test code public release → [Code Branch](https://github.com/your-username/PathVG/tree/main/code) (link will be activated on release date)  
- [ ] **2025-06-30**: Pre-trained model weights public release → [Weights Release](https://github.com/your-username/PathVG/releases/tag/weights-v1.0) (link will be activated on release date)  
- [ ] **2025-07-15**: Supplementary experiment code & visualization tools release → [Supplementary Tools](https://github.com/your-username/PathVG/tree/main/supplementary) (link will be activated on release date)  

🔧 Environment Setup
Prerequisites
Operating System: Linux (Ubuntu 20.04 recommended) / Windows 10+ (WSL2 recommended)
Python Version: 3.8 ~ 3.10
CUDA Version: 11.3 ~ 11.8 (for GPU acceleration; CPU inference is supported but not recommended)
PyTorch Version: 1.10.0 ~ 2.0.0
Installation Steps
Clone the repository:
git clone https://github.com/your-username/PathVG.git
cd PathVG

Create a virtual environment (conda recommended):
conda create -n pathvg python=3.9
conda activate pathvg

Install dependencies:
# Install PyTorch (adjust CUDA version based on your environment)
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements.txt

Verify environment (optional):
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
# Expected output: CUDA Available: True (if GPU is properly configured)

📊 Dataset (PathVG)
Overview
The PathVG dataset is the first benchmark tailored for pathology visual grounding, focusing on two core tasks:
Pathology Text-to-Region Grounding: Locate pathological regions (e.g., tumor buds, inflammatory foci) from text descriptions.
Pathology Region-to-Text Grounding: Generate text descriptions for given pathological regions.
Dataset Details
Data Source: 5,000+ hematoxylin-eosin (H&E) stained whole-slide images (WSIs) of colorectal cancer (CRC), collected from 3 clinical centers (ethics approval No.: HUST-IRB-2024-003).
Annotations:
Text annotations: 10,000+ expert-validated text descriptions (each paired with a bounding box/mask of the target region).
Region annotations: Pixel-level masks for 8 key pathological regions (tumor core, tumor invasive front, tumor buds, etc.).
Metadata: Clinical labels (e.g., TNM stage, patient age) and pathological grades (e.g., tumor bud grade).
Download & Preparation
Download the dataset from the official release link (available after 2025-05-30).
Unzip the dataset to the data/ directory (create the directory if it does not exist):
unzip PathVG_dataset_v1.0.zip -d ./data/

Verify dataset structure (should match the following):
./data/
├── train/                # Training set (3,500 WSIs + annotations)
│   ├── images/           # WSI patches (256x256, PNG format)
│   ├── masks/            # Pixel-level masks (same size as images)
│   └── annotations.json  # Text-region pairing annotations
├── val/                  # Validation set (500 WSIs + annotations)
│   └── [same structure as train/]
└── test/                 # Test set (1,000 WSIs + annotations)
    └── [same structure as train/]

Usage Notes
All WSIs are preprocessed into 256x256 patches (to balance resolution and computational efficiency).
Annotations follow the COCO format (for compatibility with mainstream grounding models).
For WSI-level inference, refer to tools/wsi_patch_process.py to split/merge patches.
🚀 Training & Testing
1. Training
Basic Training Command
Run the training script for the PathVG model (supports single-GPU and multi-GPU training):
# Single-GPU training (GPU ID: 0)
python train.py \
  --config configs/pathvg_basic.yaml \
  --gpu 0 \
  --output_dir ./results/train_basic/

# Multi-GPU training (GPU IDs: 0,1,2,3)
torchrun --nproc_per_node=4 train.py \
  --config configs/pathvg_basic.yaml \
  --gpu 0,1,2,3 \
  --output_dir ./results/train_multi_gpu/

Key Training Parameters
--config: Path to the configuration file (see configs/ for pre-defined settings, e.g., pathvg_basic.yaml for baseline, pathvg_adv.yaml for advanced model).
--gpu: GPU IDs to use (e.g., 0 for single GPU, 0,1 for two GPUs).
--output_dir: Directory to save training logs, checkpoints, and intermediate results.
--resume: Path to a checkpoint file (for resuming training, e.g., --resume ./results/train_basic/epoch_10.pth).
Training Monitoring
Logs are saved to output_dir/train.log (view in real time with tail -f ./results/train_basic/train.log).
Use TensorBoard to visualize training curves (loss, accuracy):
tensorboard --logdir ./results/train_basic/tb_logs/ --port 6006

2. Testing
Basic Testing Command
Evaluate the trained model on the PathVG test set:
python test.py \
  --config configs/pathvg_basic.yaml \
  --gpu 0 \
  --checkpoint ./results/train_basic/best_model.pth \
  --output_dir ./results/test_basic/

Key Testing Parameters
--checkpoint: Path to the pre-trained model checkpoint (use best_model.pth for the best validation performance).
--metrics: Metrics to evaluate (default: all, including R@1, R@5, IoU, BLEU-4; see utils/metrics.py for details).
--visualize: Set to True to save visualization results (bounding boxes/masks overlaid on images, e.g., --visualize True).
Test Results
Quantitative results are saved to output_dir/test_results.csv (includes metrics for each task).
Qualitative results (visualizations) are saved to output_dir/visualizations/ (if --visualize True is set).
📂 File Structure
PathVG/
├── configs/                # Configuration files for training/testing
│   ├── pathvg_basic.yaml   # Baseline model config
│   └── pathvg_adv.yaml     # Advanced model config
├── data/                   # Dataset directory (user-generated)
│   ├── train/              # Training set
│   ├── val/                # Validation set
│   └── test/               # Test set
├── models/                 # Model definitions
│   ├── pathvg.py           # Core PathVG model
│   ├── backbones/          # Backbone networks (e.g., ViT, ResNet)
│   └── grounding_heads/    # Grounding task heads (text-to-region, region-to-text)
├── tools/                  # Utility scripts
│   ├── wsi_patch_process.py# WSI patch splitting/merging
│   └── annotation_convert.py# Convert annotations to COCO format
├── utils/                  # Helper functions
│   ├── data_loader.py      # Dataset loading & augmentation
│   ├── metrics.py          # Evaluation metrics
│   └── logger.py           # Logging tools
├── train.py                # Training script
├── test.py                 # Testing script
├── requirements.txt        # Dependencies list
└── README.md               # Project documentation (this file)

📝 Citation
If you use the PathVG dataset, code, or results in your research, please cite our MICCAI 2025 paper:
@inproceedings{PathVG2025,
  title={PathVG: A New Benchmark and Dataset for Pathology Visual Grounding},
  author={Your Name, Co-Author 1, Co-Author 2, ..., Corresponding Author},
  booktitle={Proceedings of the Medical Image Computing and Computer Assisted Intervention Conference (MICCAI)},
  year={2025},
  publisher={Springer}
}

For questions or issues, please open an issue or contact the corresponding author at: [your-email@example.com]
