# EarthAlign: Learning Language-Aligned Representations of Earth System Data
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png">
</p>

<div  align="center" style="margin-top:10px;"> 

### Fadillah Adamsyah Maani<sup>1</sup>, Numan Saeed<sup>1</sup>, Jiaqi Ma<sup>1</sup>, Dat Tien Nguyen<sup>1</sup>, Youssef Wehbe<sup>2</sup>, Duncan Watson-Parris<sup>3</sup>, Salman Khan<sup>1,2</sup>

<sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), <sup>2</sup> ADIA Lab, <sup>3</sup> University of California, San Diego (UC San Diego)

</div>


[![Paper](https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white)](LINK)
[![EarthAlign_Dataset](https://img.shields.io/badge/🤗_EarthAlign-Dataset-F9D371)](#earthalign-data-preparation)
[![MeteoLIP_Weights](https://img.shields.io/badge/🤗_MeteoLIP-Weights-F9D371)](#meteolip-weights)

---

## 📢 Latest Updates
- **Oct-26**: 📜 EarthAlign paper is released [LINK].
- **Oct-26**: 🔥 MeteoLIP and weights code are released ([access](#meteolip-weights)).
- **Oct-26**: 📂 EarthAlign dataset is released ([access](#earthalign-dataset-records-and-metadata)).

---

## 🌍 Overview

**EarthAlign** is a large-scale Earth system-language alignment dataset and benchmark for learning and evaluating multimodal representations of Earth system data. It contains 553,738 samples spanning 2000-2024, with broad geographic coverage and diverse natural phenomena and normal weather conditions. Each sample includes temporal and geospatial information, multivariate ERA5 data, and four metadata-grounded caption styles that describe the same Earth system condition from different perspectives. Building on EarthAlign, **MeteoLIP** is a dual-encoder framework for aligning meteorological data with natural language. Its meteorology tokenizer, MetTok, aggregates variable-specific representations into compact meteorology tokens while incorporating temporal and geospatial information. MeteoLIP supports cross-modal retrieval, weather-condition classification, and physical variable estimation, providing a benchmark for evaluating language-aligned Earth system representations.

---

## 🏆 Highlights

- **EarthAlign Dataset:** 553,738 samples spanning 2000-2024, with broad geographic coverage and diverse natural phenomena and normal weather conditions. Four metadata-grounded caption styles provide complementary descriptions of each Earth system condition.
- **MeteoLIP:** A CLIP/SigLIP-based dual-encoder framework for meteorology-language representation learning, with MetTok for meteorological tokenization and temporal/geospatial conditioning.
- **Three Evaluation Tasks:** Cross-modal retrieval, weather-condition classification, and physical variable estimation.
- **Open Release:** Dataset, training and evaluation code, and pretrained model weights are publicly available.

---

## 🚀 Getting Started

### EarthAlign Data Preparation

1. **Download the training data.**  
   Download the training data in WebDataset format from [here](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/EarthAlign/webdatasets_train).
2. **Download the test/evaluation data.**  
   Download the test/evaluation data in `.npz` and `.json` formats from [here](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/EarthAlign/data_era5_processed), then extract the tar files.
3. **(Optional) Prepare training samples for probing.**  
   This step is only required for the classification and cyclone physical quantity estimation probing experiments. Extract the `.npz` files corresponding to the training samples from the WebDataset shards and place them in the `npz` directory of the downloaded test data, effectively combining the training and test `.npz` files in the same directory. Each `.npz` file contains a `folder_name` field, which serves as the unique sample identifier throughout the MeteoLIP pipeline.

**Note**: The EarthAlign dataset is further explained in [EarthAlign Dataset Records and Metadata](#earthalign-dataset-records-and-metadata).

### MeteoLIP Weights

We provide pretrained weights for the following MeteoLIP variants:

| Model | Meteorology Encoder | MetTok Configuration | Weights |
|---|---|---|---|
| Meteo-CLIP ViT-B | ViT-B/4 | Full MetTok | [Link](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/MeteoLIP/Meteo-CLIP/ViT-B-4_Full-MetTok) |
| Meteo-CLIP ViT-L | ViT-L/4 | Full MetTok | [Link](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/MeteoLIP/Meteo-CLIP/ViT-L-4_Full-MetTok) |
| Meteo-SigLIP ViT-B | ViT-B/4 | Full MetTok | [Link](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/MeteoLIP/Meteo-SigLIP/ViT-B-4_Full-MetTok) |
| Meteo-SigLIP ViT-L | ViT-L/4 | Full MetTok | [Link](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/MeteoLIP/Meteo-SigLIP/ViT-L-4_Full-MetTok) |

We also provide the following Meteo-CLIP ViT-B checkpoints for alternative MetTok input configurations. To use these checkpoints, adjust the corresponding constants in `meteolip/src/meteolip/utils_climate.py`:
- **Without temporal metadata input:** [Link](LINK)
- **Meteorology input only:** [Link](LINK)

### MeteoLIP Training

Train Meteo-CLIP using 4 GPUs on 1 node.

```bash
cd EarthAlign/src

torchrun --nproc_per_node=4 -m meteolip_train.main \
   --model Climate-ViT-B-4 \
   --pretrained openai \
   --dataset-type webdataset \
   --train-data '<TRAIN_DATA_DIR>/{000000000..000000070}.tar' \
   --train-num-samples 490395 \
   --val-dir-npz '<PREPROCESSED_ERA5_DATA_DIR>/npz' \
   --val-dir-cap '<PREPROCESSED_ERA5_DATA_DIR>/json' \
   --val-data '../data/test_config_samples_json/retrieval_val_samples.json' \
   --classification-val '../data/test_config_samples_json/classification_samples.json' \
   --ibtracs-wind-val '../data/test_config_samples_json/ibtracs_wind_samples.json' \
   --ibtracs-pres-val '../data/test_config_samples_json/ibtracs_pres_samples.json' \
   --lr=1e-4 \
   --wd=0.01 \
   --warmup 2000 \
   --batch-size=256 \
   --epochs=30 \
   --val-frequency 6 \
   --zeroshot-frequency 6 \
   --save-frequency 30 \
   --workers=4
```

**Notes**:
- To train ViT-L, replace `Climate-ViT-B-4` with `Climate-ViT-L-4`.
- To train a SigLIP model, use `Climate-ViT-B-4-SigLIP` or `Climate-ViT-L-4-SigLIP` as the model, set `--pretrained webli`, and add the `--siglip` flag.

### MeteoLIP Evaluation

#### Retrieval

```bash
# Options for --caption-style:
# natural_event_focused, impact_focused, physical_framing, atmospheric_pattern

cd EarthAlign/src

python eval_retrieval.py \
   --model '<MODEL_TYPE>' \
   --checkpoint '<CHECKPOINT_PATH>' \
   --val-data '../data/test_config_samples_json/retrieval_val_samples.json' \
   --val-dir-npz '<PREPROCESSED_ERA5_DATA_DIR>/npz' \
   --val-dir-cap '<PREPROCESSED_ERA5_DATA_DIR>/json' \
   --caption-style 'natural_event_focused' \
   --batch-size 64 \
   --workers 4 \
   --device cuda \
   --precision fp32 \
   --output-dir '<SAVE_DIR>' \
   --topk 5
```

#### Classification

```bash
cd EarthAlign/src

python eval_classification.py \
   --model '<MODEL_TYPE>' \
   --checkpoint '<CHECKPOINT_PATH>' \
   --classification-json-path '../data/test_config_samples_json/classification_samples.json' \
   --classification-npz-dir  '<PREPROCESSED_ERA5_DATA_DIR>/npz' \
   --workers 4 \
   --device cuda \
   --precision fp32 \
   --cm-dir '<SAVE_DIR>'
```

#### Physical Quantity Estimation

```bash
# Options for --task:
# wmo_wind, wmo_pres (adjust the --ibtracs-json-path accordingly based on the --task)

cd EarthAlign/src

python eval_phys_quant_estimation.py \
   --model '<MODEL_TYPE>' \
   --checkpoint '<CHECKPOINT_PATH>' \
   --task 'wmo_wind' \
   --ibtracs-json-path '../data/test_config_samples_json/ibtracs_{wind/pres}_samples.json' \
   --ibtracs-npz-dir '<PREPROCESSED_ERA5_DATA_DIR>/npz' \
   --batch-size 64 \
   --workers 4 \
   --device cuda \
   --precision fp32 \
   --plot-dir '<SAVE_DIR>'
```

### MeteoLIP Linear Probing

#### Classification

```bash
cd EarthAlign/src

python probing_classification.py \
   --model '<MODEL_TYPE>' \
   --checkpoint '<CHECKPOINT_PATH>' \
   --train-classification-json-path '../data/train_probing_config_samples_json/classification_samples_probing_train.json' \
   --test-classification-json-path '../data/test_config_samples_json/classification_samples.json' \
   --classification-npz-dir '<PREPROCESSED_ERA5_DATA_DIR>/npz' \
   --save-dir '<SAVE_DIR>' \
   --batch-size 64 \
   --probe-batch-size 256 \
   --epochs 50 \
   --lr 1e-3 \
   --weight-decay 0.01 \
   --workers 4 \
   --seeds 0 1 2 \
   --device cuda \
   --precision fp32
```

#### Physical Quantity Estimation

```bash
cd EarthAlign/src

python probing_phys_quant_estimation.py \
   --model '<MODEL_TYPE>' \
   --checkpoint '<CHECKPOINT_PATH>' \
   --train-wind-json-path '../data/train_probing_config_samples_json/ibtracs_wind_samples_probing_train.json' \
   --test-wind-json-path '../data/test_config_samples_json/ibtracs_wind_samples.json' \
   --train-pres-json-path '../data/train_probing_config_samples_json/ibtracs_pres_samples_probing_train.json' \
   --test-pres-json-path '../data/test_config_samples_json/ibtracs_pres_samples.json' \
   --ibtracs-npz-dir '<PREPROCESSED_ERA5_DATA_DIR>/npz' \
   --save-dir '<SAVE_DIR>' \
   --batch-size 64 \
   --probe-batch-size 256 \
   --epochs 200 \
   --lr 1e-3 \
   --weight-decay 0.01 \
   --workers 4 \
   --seeds 0 1 2 \
   --device cuda \
   --precision fp32
```

---

## 📊 EarthAlign Dataset Records and Metadata

### Natural Event Record Sources

| Source | Code | Data Link | License / Terms of Use |
|---|---|---|---|
| EM-DAT | emdat | [Link](https://zenodo.org/records/20071135) | [CC-BY-NC-ND](https://zenodo.org/records/20071135) |
| NOAA | noaa |  [Link](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | [Public Domain (U.S. Government / NOAA/NCEI)](https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf) |
| IBTrACS | ibtracs | [Link](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/) | [Open Access / WMO Resolution 40](https://community.wmo.int/site/knowledge-hub/programmes-and-initiatives/operational-information-service-ois/resolution-40) |
| COOLR | coolr | [Link](https://gpm.nasa.gov/applications/landslides/coolr) | [Open platform; see NASA COOLR data-use information](https://gpm.nasa.gov/applications/landslides/coolr) |
| PIKART | pikart | [Link](https://ar.pik-potsdam.de/?a=general) | [CC BY-NC 4.0](https://ar.pik-potsdam.de/?a=contact) |
| AusDE | ausde | [Link](https://data.gov.au/data/dataset/disaster-events-with-category-impact-and-location) | [Creative Commons Attribution 3.0 Australia (CC BY 3.0 AU)](https://creativecommons.org/licenses/by/3.0/au/) |
| MexDE | mexde | [Link](https://historico.datos.gob.mx/busca/dataset/declaratorias-sobre-emergencia-desastre-y-contingencia-climatologica/resource/946ad8a6-abe4-4ba8-b96b-7a45bfe687ea) | [LIBRE USO MX](https://historico.datos.gob.mx/busca/dataset/declaratorias-sobre-emergencia-desastre-y-contingencia-climatologica/resource/946ad8a6-abe4-4ba8-b96b-7a45bfe687ea) |
| CanDE | cdd | [Link](https://open.canada.ca/data/en/dataset/1c3d15f9-9cfa-4010-8462-0d67e493d9b9) | [Open Government Licence - Canada](https://open.canada.ca/en/open-government-licence-canada) |
| IndDE | dibi | [Link](https://dibi.bnpb.go.id/superset/dashboard/2/) | [Open Data](https://data.bnpb.go.id/fi/dataset/kompilasi-data-kejadian-dan-dampak-bencana-2025) |
| FraDE | gaspar | [Link](https://www.data.gouv.fr/datasets/base-nationale-de-gestion-assistee-des-procedures-administratives-relatives-aux-risques-gaspar) | [Licence Ouverte / Open Licence](https://www.data.gouv.fr/datasets/base-nationale-de-gestion-assistee-des-procedures-administratives-relatives-aux-risques-gaspar) |

### Dataset Record JSON

The complete `data/record_json` dataset is provided separately at [here](https://huggingface.co/datasets/MBZUAI/EarthAlign/tree/main/EarthAlign/data/record_json). The folder contains JSON files corresponding to the different natural event record sources, as well as climatology-based weather normal (`negsamp`) samples. ERA5 data are provided under [CC BY](https://confluence.ecmwf.int/spaces/CKB/pages/222493847/Use%2BCase%2B2%2BERA5%2Bhourly%2Bdata%2Bon%2Bsingle%2Blevels%2Bfrom%2B1940%2Bto%2Bpresent?desktop=true&macroName=page-info).

Each JSON file contains a list of dictionaries with the following fields:

- `ERA5_FOLDER_NAME` — Unique sample identifier used throughout the pipeline.
- `ERA5_SNAPSHOT_TIME` — Timestamp of the ERA5 snapshot retrieved for the sample.
- `ERA5_SNAPSHOT_BBOX` — Bounding box of the ERA5 snapshot retrieved for the sample.
- `EVENT_CATEGORIES` — Event category or categories associated with the sample. A sample may have multiple categories, depending on the original source.
- `EVENT_YEARS` — Year or years in which the event occurred.
- `CAPTIONS` — Multi-style captions generated for the sample.
- `CLEANED_METADATA` — Cleaned metadata used for grounded multi-style caption generation. This field is not provided for EM-DAT records due to the source's license restrictions (the corresponding metadata can be obtained from the original EM-DAT records).
- `METADATA` — Metadata associated with the original record. For EM-DAT records, only the `DisNo.`, preprocessed `locations`, and `bbox` are provided. The `DisNo.` can be used to retrieve the corresponding original EM-DAT record and its metadata. A `DisNo.` containing `---` indicates an upsampled sample; see the EarthAlign paper for details.

### JSON for Testing/Evaluation and Probing

The `data/test_config_samples_json` folder contains JSON files used to define samples for testing/evaluation. Each JSON file contains a list with the following details:
- `classification_samples.json` — Each sample contains its unique sample identifier, standardized event class, and the year or years in which the event occurred.
- `ibtracs_pres_samples.json` — Each sample contains its unique sample identifier, standardized cyclone class or subclass, and the ground-truth minimum central pressure.
- `ibtracs_wind_samples.json` — Each sample contains its unique sample identifier, standardized cyclone class or subclass, and the ground-truth maximum sustained wind.
- `retrieval_val_samples.json` — Each sample contains its unique sample identifier and the year or years in which the event occurred.

The `data/train_probing_config_samples_json` folder contains JSON files used to define the training samples for probing experiments. Each JSON file contains a list with the following details:
- `classification_samples_probing_train.json` — Same fields as `classification_samples.json`.
- `ibtracs_pres_samples_probing_train.json` — Same fields as `ibtracs_pres_samples.json`.
- `ibtracs_wind_samples_probing_train.json` — Same fields as `ibtracs_wind_samples.json`.

---

## 🙏 Acknowledgement

This material is based on work supported by ADIA Lab, Abu Dhabi under a collaborative research project titled Climate Co-Pilot: A Multimodal AI Agent for Climate Resilience in the MENA Region.

Our codebase is built upon the [OpenCLIP](https://github.com/mlfoundations/open_clip) repository, and we thank sincerely its developers and contributors for making their work publicly available.


---

## 📜 Citation

CITATION

---
<div style="display: flex; align-items: center; gap: 10px;">
  <a href="https://www.ival-mbzuai.net">
    <img src="assets/logos/IVAL_logo.png" height="100">
  </a>
  <a href="https://mbzuai.ac.ae">
    <img src="assets/logos/MBZUAI_logo.png" height="85">
  </a>
  <a href="https://www.ucsd.edu/">
    <img src="assets/logos/UCSD_logo.svg" height="85">
  </a>
  <a href="https://www.adialab.ae/">
    <img src="assets/logos/ADIALab_logo.png" height="40">
  </a>
</div>