\# Shrimp Disease Detection Using Deep Learning



\## Project Overview



This project uses deep learning to classify shrimp images into four categories: Healthy, Black Gill (BG), White Spot Syndrome Virus (WSSV), and combined WSSV and Black Gill (WSSV\_BG).



The goal is to support research into image-based shrimp disease classification in aquaculture.



\## Model Architecture



\* ConvNeXt-Tiny as the feature extraction backbone

\* Convolutional Block Attention Module (CBAM)

\* Focal Loss for model training

\* Grad-CAM++ for visualizing regions influencing predictions



\## Dataset



The project uses the ShrimpDiseaseBD dataset. The images are organized into four classes:



\* BG — Black Gill

\* Healthy — healthy shrimp

\* WSSV — White Spot Syndrome Virus

\* WSSV\_BG — combined WSSV and Black Gill



The dataset is divided into training, validation, and testing sets. Source-group separation is used to reduce the risk of related images appearing across multiple splits.



\## Final Test Results



Evaluation was performed on 189 test images.



| Metric             | Result |

| ------------------ | -----: |

| Accuracy           | 78.31% |

| Weighted precision | 78.81% |

| Weighted recall    | 78.31% |

| Weighted F1-score  | 78.40% |



The model performs best on the Healthy class. Confusion between WSSV and WSSV\_BG remains a limitation.



\## Explainability



Grad-CAM++ visualizations are generated to inspect image regions influencing model predictions. These visualizations are useful for analysis but do not establish clinical or diagnostic reliability.



\## Repository Structure



\* `src/` — dataset preparation, model training, evaluation, and explainability scripts

\* `data/splits/` — dataset split CSV files

\* `outputs/` — local evaluation results, figures, and model checkpoints



Dataset images and model checkpoints are excluded from Git tracking.



\## Limitations



The model is a research prototype evaluated on the available dataset. Its performance on new images, other farms, or different imaging conditions has not been established. Predictions should not replace assessment by an aquaculture or veterinary expert.



\## Team



\* Gutlapalli Vyshali

\* Kasarapu Yogini satya divya

\* Kankanala Amrutha

\* Gedela Varshitha



