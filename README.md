# Whale Track - Telemetry-Driven Georeferenced Multi-Whale Tracking from Drone Video

![Logo](assets/whaletrack_pin.svg)
_WhaleTrack_, is a modular system that combines oriented-bounding-box detection, multi-object tracking with lightweight spatial relinking, telemetry-driven ray–plane reprojection from drone GPS, altitude, and gimbal orientation, and per-frame field-of-view estimation.

This repository contains the code accompagnying a paper presented at the 2nd Workshop on Marine Vision in conjunction with the _19th European Conference on Computer Vision_ -- ECCV 2026

License : see LICENSE file

## WhaleTrack installation

#### Installation with `pixi`(recommended)

> [!TIP]
> If you neeed to install `pixi`, please visit the [Pixi Installation Guide](https://pixi.prefix.dev/latest/installation/).

````bash
pixi install`
``̀

```bash
pixi run python src/whaletrack/reprojection_to_world.py
``̀

### Installation with `pip`

```bash
python -m venv .venv
source .venv/bin/activate # Linux/Mac
# .venv\Scripts\activate # Windows
pip install -e .
````

## Get the dataset

To run the WhaleTrack detection, tracking, and georeferencing pipeline you need to get :

- the Whale Drone dataset available at https://huggingface.co/datasets/LucieLprt-Dvldr/WhaleDrone
- and the YOLO26n oriented bounding-box (OBB) model fine-tuned on this dataset at https://huggingface.co/pierreadorni/WhaleDrone


There is a python utility command to get both provide with the code.
You can run it either with pixi or with python.

With `pixi`:

```
pixi run get_dataset --minimal
```

or inside your virtual environment :

```
python get_dataset.py --minimal
```

The `--minimal` option enable to download a minimal subset of the dataset in order to test the code. You can remove it to download the full dataset which is about 135GB.

The dataset would then be download in a 'dataset' subfolder.
