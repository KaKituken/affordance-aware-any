# Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion

## 🔨 Installation
Run the following instruction to set up the environment.

```sh
cd model
conda env create -f environment.yaml
```

## 🏷️ Dataset

### SAM-FB Dataset

We build the SAM-FB dataset on top of original [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset.

Due to the storage limitation, you can download the first 10 sub-folder of SAM-FB dataset [here](). 

To build the rest of the dataset, please download the [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset and follow the instructions in the next section.

### Build Your Own Dataset

Our automatic pipeline allows user to build their own dataset by providing only the images. You can either download the original [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset or provide a set of images under `./data` folder.

#### Download SAM
First, download the SAM checkpoint from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put it int the `./ckpt` folder.

#### Get the masks
Saying you have a folder from SA-1B, `./data/sa_000010/`

Please run the following instruction to get the mask after NMS
```sh
cd dataset-builder/img_data

python mask_nms.py --image-dir ./data/sa_000010/ --save-dir ./data/sa_000010/ --model-path ./ckpt/sam_vit_h_4b8939.pth
```

#### Inpainting
After getting the masks, run following to inpaint the background and construct pair data. The filter will also perform here.

Download the checkpoint [here]() for foreground filtering.

```sh
python inpainting_sa1b.py --annotation-path ./data/sa_000010/ --image-path ./data/sa_000010/ --save-path ./SAM-FB/ --model-path ./ckpt/quality_ctrl.pth
```

### Video Dataset (Youtube-VOS)
We also support video dataset. For [YTB-VOS]()-like dataset, simply run

```sh
cd dataset-builder/video_data

python vdata_process.py -a ./data/Youtube-VOS/train/Annotations -i ./data/Youtube-VOS/train/JPEGImages -s ./SAM-FB-video --start-index 11
```