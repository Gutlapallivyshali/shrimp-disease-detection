\# Shrimp Disease Detection - Experiment Results



\## Dataset

ShrimpDiseaseDB



\## Classes

\- BG

\- Healthy

\- WSSV

\- WSSV\_BG



\---



\## Experiment 1: ConvNeXt-Tiny Baseline



Model:

\- ConvNeXt-Tiny

\- Pretrained ImageNet weights

\- CrossEntropyLoss

\- AdamW

\- Learning rate: 0.0001

\- Epochs: 10



Test Results:

\- Accuracy: 74.60%

\- Precision: 77.42%

\- Recall: 74.60%

\- Weighted F1: 74.79%



\---



\## Experiment 2: ConvNeXt-Tiny + CBAM + Focal Loss



Model:

\- ConvNeXt-Tiny

\- CBAM attention

\- Focal Loss

\- AdamW

\- Learning rate: 0.0001

\- Epochs: 10



Best Recorded Test Results:

\- Accuracy: 76.19%

\- Precision: 77.12%

\- Recall: 76.19%

\- Weighted F1: 76.46%



This is the best recorded test result.



\---



\## Experiment 3: CBAM + Focal Loss + Scheduler



Model:

\- ConvNeXt-Tiny

\- CBAM

\- Focal Loss

\- AdamW

\- Initial learning rate: 0.0001

\- ReduceLROnPlateau scheduler

\- Epochs: 10



Results:

\- Best Validation Accuracy: 80.12%

\- Test Accuracy: 68.78%

\- Weighted F1: 69.85%



Conclusion:

The high validation accuracy did not translate into better test performance.



\---



\## Experiment 4: CBAM + Focal Loss + Frozen Backbone



Model:

\- ConvNeXt-Tiny

\- CBAM

\- Focal Loss

\- ConvNeXt feature extractor frozen

\- AdamW

\- Learning rate: 0.0001

\- Epochs: 10



Best Validation Accuracy:

\- 60.23%



Conclusion:

Freezing the complete backbone was too restrictive.



The model was not evaluated on the test set.



\---



\## Experiment 5: CBAM + Focal Loss + Lower Learning Rate



Model:

\- ConvNeXt-Tiny

\- CBAM

\- Focal Loss

\- AdamW

\- Learning rate: 0.00003

\- Epochs: 10



Best Validation Accuracy:

\- 77.19%



Test Results:

\- Accuracy: 71.43%

\- Precision: 74.21%

\- Recall: 71.43%

\- Weighted F1: 71.67%



Conclusion:

The lower learning rate did not improve test performance.



\---



\## Current Best Recorded Model



Architecture:

ConvNeXt-Tiny + CBAM + Focal Loss



Best Test Accuracy:

76.19%



Best Weighted F1:

76.46%



Note:

The exact checkpoint corresponding to the 76.19% experiment was overwritten by a later scheduler experiment. Therefore, the current file named

convnext\_tiny\_cbam\_focal\_best.pth

must not be treated as the 76.19% checkpoint.

