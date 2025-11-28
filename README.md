# 💇‍♀️ DiffLocks Studio

**AI-powered 3D hair generation made easy.**

This repository is a non-official fork of [Meshcapade/DiffLocks](https://github.com/Meshcapade/difflocks), refactored to be accessible, stable, and ready for cloud deployment.
(Under development, currently only the kaggle notebook works, in a private way)

### 🚀 Key Features
*   **Interactive UI:** Full Gradio interface—no coding required to generate hair.
*   **Cloud Ready:** Optimized to run on **Kaggle** and **Google Colab** (Free Tier T4 GPUs).
*   **Stability Fixes:** Implements FP32 inference and VRAM cleanup to prevent crashes.
*   **Export Tools:** Auto-generates `.OBJ` files and Blender import scripts.

*   ## 💻 Local Installation

You can run DiffLocks Studio on your own computer if you have an **NVIDIA GPU**.

### Prerequisites
*   **OS:** Windows 10/11 or Linux.
*   **GPU:** NVIDIA GeForce card (GTX 1060 or better recommended) with drivers installed.
*   **Software:**
    *   [Python 3.10 or 3.11](https://www.python.org/downloads/) (Make sure to tick **"Add Python to PATH"** during installation).
    *   [Git](https://git-scm.com/downloads).

### Quick Start (Windows) (Under tests)

1.  **Download:** Click the green **Code** button above and select **Download ZIP**. Extract it to a folder.
    *   *Alternative:* Open a terminal and run `git clone https://github.com/arqdariogomez/difflocks.git`
2.  **Run:** Double-click on **`run_windows.bat`**.
3.  **Wait:** The first run will download necessary libraries (PyTorch, etc.). This might take 5-10 minutes.
4.  **Use:** Once finished, it will provide a local URL (usually `http://127.0.0.1:7860`). Open that link in your browser.

### Quick Start (Linux) (Under tests)

1.  Clone the repository:
    ```bash
    git clone https://github.com/arqdariogomez/difflocks.git
    cd difflocks
    ```
2.  Run the launcher:
    ```bash
    ./run_linux.sh
    ```

Original readme starts here:

# DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models #

[**Paper**](https://arxiv.org/abs/2505.06166) | [**Project Page**](https://radualexandru.github.io/difflocks/)

<p align="middle">
  <img src="imgs/teaser.png" width="650"/>
</p>

This repository contains official inference and training code for DiffLocks, which creates strand-based realistic hairstyle from a single image. It also contains the DiffLocks dataset consisting of 40K 3D synthetic strand-based hair data generated in Blender.

## Requirements 

DiffLocks dependencies can be installed from the provided `requirements.txt` which can be installed in a virtual environment: 

	$ python3 -m venv ./difflocks_env
	$ source ./difflocks_env/bin/activate
    $ pip install -r ./requirements.txt

Afterwards we need to install custom CUDA kernels for the diffusion model:
* [NATTEN](https://github.com/SHI-Labs/NATTEN/tree/main) for the sparse (neighborhood) attention used at low levels of the hierarchy.
* [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) for global attention.
Please double check that you install Natten for torch 2.5.0 (as per requierments.txt).

Finally if you want to perform inference, you need to download the checkpoints for the trained models. 
The pretrained checkpoints can be downloaded by following [this section](#download-pretrained-checkpoints):

    

## Dataset
![data](/imgs/dataset.png "dataset")
The DiffLocks dataset consists 40K hairstyle. Each sample includes 3D hair (~100K strands), corresponding rendered RGB image and metadata regarding the hair. 

DiffLocks can be downloaded using: 

    ./download_dataset.sh <DATASET_PATH_CONTAINING_ZIPPED_FILES>

After downloading, the dataset has to first be uncompressed:

    $ ./data_processing/uncompress_data.py --dataset_zipped_path <DATASET_PATH_CONTAINING_ZIPPED_FILES> --out_path <DATASET_PATH>

After uncompressing we create a processed dataset:

	$ ./data_processing/create_chunked_strands.py --dataset_path <DATASET_PATH>
	$ ./data_processing/create_latents.py --dataset_path=<DATASET_PATH> --out_path <DATASET_PATH_PROCESSED>
	$ ./data_processing/create_scalp_textures.py --dataset_path=<DATASET_PATH> --out_path <DATASET_PATH_PROCESSED> --path_strand_vae_model ./checkpoints/strand_vae/strand_codec.pt


## Download pretrained checkpoints
You can download pretrained checkpoints by running:

	./download_checkpoints.sh

## Inference
To run inference on an RGB and create 3D strands use:

    $ ./inference_difflocks.py \
		--img_path=./samples/medium_11.png \
		--out_path=./outputs_inference/ 

You also have options to export a `.blend` file and an alembic file by specifying `--blender_path` and `--export_alembic` in the above script. 
Note that the blender path corresponds to the blender executable with version 4.1.1. It will likely not work with other versions. 

	
## Train StrandVAE 
To train the strandVAE model: 

	$ ./train_strandsVAE.py --dataset_path=<DATASET_PATH> --exp_info=<EXP_NAME>

it will start training and outputting tensorboard logs in `./tensorboard_logs`


## Train DiffLocks diffusion model 
To train the diffusion model: 

	$ ./train_scalp_diffusion.py \
		--config ./configs/config_scalp_texture_conditional.json \
		--batch-size 4 \
		--grad-accum-steps 4 \
		--mixed-precision bf16 \
		--use-tensorboard \
		--save-checkpoints \
		--save-every 100000 \
		--compile \
		--dataset_path=<DATASET_PATH> \
		--dataset_processed_path=<DATASET_PATH_PROCESSED>
		--name <EXP_NAME> 

it will start training and outputting tensorboard logs in `./tensorboard_logs`. 
Start training on multiple GPUs by first running:

	$ accelerate config

followed by pre-pending `accelerate launch` to the previous training script:

	$ accelerate launch ./train_scalp_diffusion.py \
		--config ./configs/config_scalp_texture_conditional.json \
		--batch-size 4 \
		<ALL_THE_OTHER_OPTIONS_AS_SPECIFIED_ABOVE>

You would probably to adjust the `batch-size` and `grad-accum-step` depending on the number of GPUs you have. 



 






