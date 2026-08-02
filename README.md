
## Overview

**CPE-MVQA** is the official implementation of:

> **CPE-MVQA: Counterfactual Pathway Regulation for Mitigating Language Shortcuts in Medical Visual Question Answering**

<p style="line-height:1.7; text-align:justify;">   Medical Visual Question Answering (Med-VQA) aims to answer clinically relevant questions by jointly reasoning over medical images and textual queries. However, existing Med-VQA models often suffer from language shortcuts, where models rely excessively on textual correlations while insufficiently utilizing relevant visual information.  To address this issue, we propose CPE-MVQA (Counterfactual Pathway Regulation for Mitigating Language Shortcuts in Medical Visual Question Answering), a pathway-level counterfactual framework that models multimodal reasoning from the perspective of information propagation pathways. CPE-MVQA estimates pathway-specific responses through counterfactual interventions and adaptively regulates the competition between visual and textual pathways during reasoning.  The framework introduces an Offline Counterfactual Path Bank (OCPB) to construct sample-specific pathway priors and employs Causal Path Regulation (CPR) and Causal-Semantic Routing (CSR) to regulate cross-modal information propagation while preserving useful semantic information.  By combining pathway-aware counterfactual modeling with adaptive cross-modal regulation, CPE-MVQA reduces language shortcut bias and improves multimodal reasoning in Med-VQA. Experiments on public benchmarks, including SLAKE and PathVQA, demonstrate the effectiveness of the proposed framework. </p>

<p align="center">
<img src="imgs/main_structure.jpg" width="90%">
</p>



## Key Features

- Pathway-level counterfactual modeling for Med-VQA reasoning. 
- Offline Counterfactual Path Bank (OCPB) for pathway prior construction. 
- Causal Path Regulation (CPR) for adaptive cross-modal regulation. 
- Causal-Semantic Routing (CSR) for semantic preservation.

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/cloneiq/CPE-MVQA.git
cd CPE-MVQA
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Prepare Datasets and Pretrained Weights

Prepare the datasets, pretrained weights, and `roberta-base` files according to the instructions in [Data Preparation](#data-preparation).

### Train and Test

Run training and testing scripts as described in [Train & Test](#train--test).

## Project Structure

```bash
CPE-MVQA/
├── checkpoints/
├── data/
│   ├── slake/
│   │   ├── imgs/
│   │   ├── train.json
│   │   ├── valid.json
│   │   └── test.json
│   ├── pathvqa/
│   │   └── ...
├── pretrained_weights/
│   ├── m3ae.ckpt
├── roberta-base/
├── pipeline/              #Causal Response Computation Module
│   ├──...
├── interventions/         # Counterfactual intervention modules
│   ├──...
├── build_causal_cache.py  #Causal Cache Module
├── scripts/         
├── main.py
├── train.py
└── test.py
```

## Data Preparation

### Datasets

Please download the following datasets and place the files under the `data/` directory.

> | Dataset | Description | Download |
> |---|---|---|
> | SLAKE | An English-Chinese bilingual Med-VQA benchmark containing 642 radiology images, including CT, MRI, and X-ray images, and 14,028 question-answer pairs, plus pixel-level masks and a medical knowledge graph. | [SLAKE](https://www.med-vqa.com/slake/) |
> | PathVQA | PathVQA is a large-scale medical visual question answering benchmark for pathology image understanding. It contains 4,998 pathology images with 32,799 question-answer pairs, covering both open-ended and closed-ended questions for multimodal reasoning evaluation. | [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
>

### Pretrained Weights

Download the **M3AE pretrained weight** and put it in the `/pretrained_weights` directory:

- [M3AE pretrained weight](https://drive.google.com/drive/folders/1b3_kiSHH8khOQaa7pPiX_ZQnUIBxeWWn)

### roberta-base

Download `roberta-base` and put it in the `/roberta-base` directory:

- [roberta-base](https://drive.google.com/drive/folders/1ouRx5ZAi98LuS6QyT3hHim9Uh7R1YY1H)

## Train & Test

```bash
# Train
python main.py
# Test
python test.py
```

## Results

### Results on  SLAKE and PathVQA

| **Method** | **Reference** | SLAKE-Overall | SLAKE-Open | SLAKE-Closed | PathVQA-Overall | PathVQA-Open | PathVQA-Closed |
| ---------- | ------------- | ------------- | ---------- | ------------ | --------------- | ------------ | -------------- |
| MEVF-BAN   | MICCAI’19     | 77.66         | 75.19      | 81.49        | 44.85           | 8.10         | 81.40          |
| MEVF-SAN   | MICCAI’19     | 75.87         | 74.57      | 77.88        | 43.60           | 6.00         | 81.00          |
| BiRL       | JBI’22        |               |            |              | 54.34           | 22.82        | 85.58          |
| VQAMix     | TMI’22        |               |            |              | 48.60           | 13.40        | 83.50          |
| AMAM       | KBS’22        |               |            |              | 50.40           | 18.20        | 84.40          |
| M3AE       | MICCAI’22     | 84.83         | 83.34      | 87.08        | 59.98           | 30.86        | 88.91          |
| PubMedCLIP | EACL’23       | 80.10         | 78.40      | 82.50        |                 |              |                |
| CPCR       | TMI’23        | 81.90         | 80.50      | 84.10        |                 |              |                |
| M2I2       | ISBI’23       | 81.20         | 74.70      | **91.10**    | 62.20           | **36.30**    | 88.00          |
| CCIS-MVQA  | TMI’24        | 84.08         | 80.12      | 86.72        |                 |              |                |
| UnICLAM    | MedIA’25      | 83.10         | 81.10      | 85.70        |                 |              |                |
| BaMCo      | MICCAI’25     | 85.80         | 84.20      | 87.30        | 60.00           | 29.10        | **90.80**      |
| CPE-MVQA   |               | **86.10**     | **85.37**  | 87.20        | **62.65**       | 35.67        | 89.47          |

## Acknowledgement

Our project references the code in the following repository. Thanks for their work and sharing.

- [M3AE](https://github.com/zhjohnchan/M3AE)

## Future Work



- Developing more fine-grained and robust pathway-response modeling strategies. 
- Exploring end-to-end dynamic learning of pathway responses to reduce offline preprocessing dependency.
- Extending the framework to additional medical modalities and real-world clinical scenarios for improved generalization and applicability.

## Citation

If you use this code for your research, please cite our paper.

## Contact

Yao Tang, Kunming University of Science and Technology Kunming, Yunnan CHINA, email: yao43065@gmail.com

Lijun Liu, Associate Professor (Ph.D.), Kunming University of Science and Technology Kunming, Yunnan CHINA, email: cloneiq@kust.edu.cn

