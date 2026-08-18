# Reliability Assessment of Deep Learning and Explainability Methods for Face Anti-Spoofing

**Author:** ........................................................  
**Institution:** ........................................................

## Abstract

Face anti-spoofing is an important component of biometric authentication systems. In addition to distinguishing bona fide faces from presentation attacks, a reliable model should remain stable under image-quality changes and provide meaningful explanatory evidence. This study evaluates a lightweight deep-learning model on the CelebA-Spoof dataset and compares two post-hoc explanation methods, Grad-CAM and Integrated Gradients. The experiment uses 10,000 training images, 1,000 validation images, and 1,000 test images. The model achieves an area under the receiver operating characteristic curve of 0.9446 and an average precision of 0.9526. However, at the decision point selected from the validation set, the false acceptance rate for spoof samples remains 46.2%, indicating that strong ranking performance does not necessarily produce a safe operating point. Grad-CAM is more stable than Integrated Gradients across all spatial-similarity measures. The findings support a multidimensional evaluation protocol in which classification performance, robustness, explanation stability, faithfulness, and dataset limitations are considered jointly.

**Index Terms—** face anti-spoofing, deep learning, model explainability, Grad-CAM, Integrated Gradients, CelebA-Spoof, reliability.

## I. INTRODUCTION

Face authentication systems may be deceived by printed photographs, screen displays, masks, or other presentation attacks. Face anti-spoofing is therefore not merely an image-classification problem; it is also a risk-assessment problem under changing data conditions.

Prediction accuracy alone is insufficient to establish model reliability. A model may perform well on a test set while relying on auxiliary cues such as compression artifacts, illumination conditions, or background characteristics. This study evaluates classification performance, prediction stability, explanation stability, faithfulness, sanity behavior, and computational cost within a single controlled protocol.

## II. THEORETICAL BACKGROUND

### A. Face Anti-Spoofing

Face anti-spoofing determines whether an input image represents a bona fide face or a presentation attack. Attack types differ in material, texture, light reflection, and spatial relationship between the face and its environment. A generalizable model should therefore learn face-related evidence rather than memorize a limited set of dataset-specific patterns.

### B. Deep-Model Explainability

Grad-CAM produces a saliency map using gradient information from deep convolutional layers and identifies spatial regions associated with the model decision. Integrated Gradients estimates the contribution of individual pixels by accumulating gradients along a path from a reference image to the input image. Explanation stability measures whether a map is preserved after an image transformation. Faithfulness measures how the prediction changes when important regions are removed or restored. Sanity evaluation examines whether explanations respond when model parameters are randomized.

## III. RESEARCH METHODOLOGY

### A. Dataset and Experimental Design

The study uses CelebA-Spoof while preserving the official separation between training and testing data. The training set contains 10,000 images, the validation set contains 1,000 images, and the test set contains 1,000 images. The training and test sets are balanced by class. Images are converted to RGB, resized to 224 x 224 pixels, and normalized according to the convention used by the pretrained model.

Three image-quality transformations are evaluated: brightness reduction, Gaussian blur, and JPEG compression. Each transformation is applied at three increasing severity levels. These transformations are excluded from training so that the evaluation measures robustness to previously unseen quality changes.

### B. Model and Training Procedure

The model is based on MobileNetV3-Small, a convolutional architecture with relatively low computational cost. Its initial representation is transferred from training on a large natural-image corpus, and the final classifier is adapted to distinguish bona fide and spoof samples.

Training is conducted in two stages. The first stage updates only the classification component, while the second stage fine-tunes the final three feature blocks. The total training duration is six epochs with a batch size of 32. AdamW optimization, weight decay, gradient clipping, and mixed-precision computation are used. The best model is selected according to ranking performance on the validation set, and the resulting decision point is fixed before test evaluation.

### C. Explainability Evaluation

Map similarity is measured using cosine similarity, Spearman rank correlation, and overlap among the most salient regions. Faithfulness is evaluated by progressively removing important regions and by progressively restoring them from a reference image. Sanity evaluation is performed by progressively randomizing parts of the model. Runtime is measured separately for prediction, explanation generation, removal, and restoration procedures.

## IV. EXPERIMENTAL RESULTS

### A. Classification Performance

| Metric | Result | 95% Confidence Interval |
|---|---:|---:|
| Accuracy | 0.766 | [0.738, 0.792] |
| Precision | 0.989 | [0.975, 1.000] |
| Spoof recall | 0.538 | [0.493, 0.579] |
| F1-score | 0.697 | [0.658, 0.730] |
| ROC area | 0.945 | [0.930, 0.957] |
| Average precision | 0.953 | — |
| Spoof false acceptance rate | 0.462 | [0.421, 0.507] |
| Bona fide false rejection rate | 0.006 | [0.000, 0.014] |
| Average classification error | 0.234 | [0.214, 0.257] |

The confusion matrix shows that 497 of 500 bona fide images are correctly recognized, whereas only 269 of 500 spoof images are detected. The model is therefore conservative toward bona fide samples but insufficiently sensitive to spoof samples. A diagnostic optimization performed directly on the test set reduces the average classification error to 0.114; this value is not used as the final estimate because it is optimized on the evaluation data.

### B. Prediction Stability

| Transformation | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|
| Blur | 0.985 | 0.931 | 0.846 |
| Brightness reduction | 0.977 | 0.960 | 0.941 |
| JPEG compression | 0.840 | 0.815 | 0.774 |

JPEG compression has the largest effect on classification decisions, whereas brightness reduction has the smallest effect. Stability decreases as transformation severity increases for all three transformation families.

### C. Explanation Stability

| Method | Cosine Similarity | Spearman Correlation | Top-10% Overlap | Top-20% Overlap |
|---|---:|---:|---:|---:|
| Grad-CAM | 0.9119 | 0.8186 | 0.5608 | 0.6236 |
| Integrated Gradients | 0.7098 | 0.5580 | 0.3123 | 0.3770 |

The results are computed over 2,420 pairs with unchanged classification decisions. Grad-CAM achieves higher values than Integrated Gradients across all four measures. This advantage is maintained as the severity of the image transformation increases.

### D. Faithfulness, Sanity, and Runtime

Table I reports the faithfulness results for 50 images per method and reference-image condition. Lower removal area and higher restoration area are preferred under the evaluation protocol.

| Explanation Method | Reference Image | Removal Area | Restoration Area |
|---|---|---:|---:|
| Grad-CAM | Mean-color reference | 0.606 | 0.821 |
| Grad-CAM | Blurred reference | 0.734 | 0.908 |
| Integrated Gradients | Mean-color reference | 0.690 | 0.695 |
| Integrated Gradients | Blurred reference | 0.804 | 0.816 |

Grad-CAM achieves lower removal areas and higher restoration areas than Integrated Gradients in both reference-image conditions. The results vary according to the reference image; therefore, they should not be interpreted as direct causal evidence.

When the model is randomized at the highest level, map similarity approaches zero. This indicates that the explanations depend on model parameters, but it does not prove that the highlighted regions are the actual causes of the predictions.

| Explanation Method | Randomization Level | Cosine Similarity | Spearman Correlation | Top-10% Overlap |
|---|---:|---:|---:|---:|
| Grad-CAM | 1 | 0.3863 | -0.0338 | 0.0832 |
| Grad-CAM | 2 | 0.5428 | 0.2486 | 0.1381 |
| Grad-CAM | 3 | 0.0000 | 0.0000 | 0.0382 |
| Integrated Gradients | 1 | 0.6714 | 0.5498 | 0.2518 |
| Integrated Gradients | 2 | 0.6548 | 0.5440 | 0.2501 |
| Integrated Gradients | 3 | 0.0000 | 0.0000 | 0.0553 |

**TABLE I. FAITHFULNESS AND SANITY RESULTS.**

| Operation | Mean (ms) | Median (ms) | Standard Deviation (ms) | Samples |
|---|---:|---:|---:|---:|
| Classifier prediction | 6.32 | 6.32 | 0.03 | 50 |
| Grad-CAM | 7.62 | 7.62 | 0.05 | 50 |
| Integrated Gradients | 47.99 | 47.98 | 0.09 | 50 |
| Removal evaluation | 13.47 | 13.47 | 0.08 | 50 |
| Restoration evaluation | 95.97 | 95.96 | 0.12 | 50 |

On an NVIDIA Tesla T4 GPU, the median time for one prediction is 6.32 ms, for Grad-CAM is 7.62 ms, and for Integrated Gradients is 47.98 ms. The removal test requires 13.47 ms, whereas the restoration test requires 95.96 ms.

### E. Error by Spoof Type

Table II reports the false acceptance rate and spoof detection rate by attack type. The reported sample counts are the number of test samples available for each category.

| Spoof Type | Samples | False Acceptance Rate | Spoof Detection Rate | Mean Spoof Score |
|---|---:|---:|---:|---:|
| A4 | 76 | 0.513 | 0.487 | 0.758 |
| PC | 56 | 1.000 | 0.000 | 0.117 |
| Face mask | 42 | 0.214 | 0.786 | 0.911 |
| Pad | 37 | 0.081 | 0.919 | 0.997 |
| Phone | 42 | 0.190 | 0.810 | 0.963 |
| Photo | 40 | 0.175 | 0.825 | 0.861 |
| Poster | 62 | 0.548 | 0.452 | 0.777 |
| Region mask | 34 | 0.118 | 0.882 | 0.973 |
| Three-dimensional mask | 43 | 0.140 | 0.860 | 0.968 |
| Upper-body mask | 68 | 0.956 | 0.044 | 0.379 |

**TABLE II. ERROR BY SPOOF TYPE.**

The highest false acceptance rates occur for PC, upper-body mask, poster, and A4 attacks. Pad, region-mask, and three-dimensional-mask attacks are detected more successfully. This difference indicates that performance depends strongly on attack material, presentation format, and the distance between training and test domains.

## V. DISCUSSION

The model provides relatively strong ranking performance but does not yet provide a safe operating point. This distinction is important in anti-spoofing systems: the ROC area summarizes ranking behavior across many decision points, whereas a deployed system operates at one selected point. The decision point should therefore be selected according to the cost of accepting a spoof and rejecting a bona fide user.

Grad-CAM is more stable than Integrated Gradients in all reported tests. A plausible interpretation is that Grad-CAM produces a more spatially concentrated signal, whereas Integrated Gradients is more sensitive to local pixel changes and to the reference image. This is an interpretation of the observed results, not evidence of a causal mechanism.

## VI. LIMITATIONS

The experiment uses a relatively small subset and a single primary architecture. Subject overlap may exist across data partitions, limiting the evaluation of generalization to unseen subjects. The study also contains one official run and does not evaluate cross-dataset transfer, real-camera conditions, demographic fairness, or previously unseen attack types.

Removal and restoration tests may generate images outside the natural data distribution. Their faithfulness scores should therefore be treated as empirical indicators. Stronger conclusions require subject-disjoint partitioning, multiple independent runs, and evaluation on additional datasets.

## VII. CONCLUSION

This study evaluates a lightweight deep-learning model for face anti-spoofing together with two post-hoc explanation methods. The model achieves strong ranking performance, but the spoof miss rate remains high at the selected operating point. Grad-CAM is more stable than Integrated Gradients under image-quality transformations and requires less computation. Nevertheless, stability, faithfulness, and sanity behavior should not be interpreted as proof of causality or absolute security.

## REFERENCES

[1] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proc. IEEE Int. Conf. Comput. Vis.*, 2017, pp. 618–626.

[2] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in *Proc. Int. Conf. Mach. Learn.*, 2017, pp. 3319–3328.

[3] Y. Zhang *et al.*, “CelebA-Spoof: Large-scale face anti-spoofing dataset with rich annotations,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2020, pp. 1549–1558.

[4] Experimental results, dataset manifests, and environment records from the XAI Face Anti-Spoofing MVP project.
