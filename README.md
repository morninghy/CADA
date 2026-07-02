# CADA

Class-Aware Domain Adaptive Object Detection via Dynamic Cross-Domain Confusion Modeling

This repository provides the implementation of **CADA: Class-Aware Domain Adaptive Object Detection via Dynamic Cross-Domain Confusion Modeling**. The project focuses on unsupervised domain-adaptive object detection in autonomous driving scenarios. It specifically addresses class imbalance in real-world road data, where minority classes have fewer samples and weaker feature boundaries, making them prone to semantic drift toward visually similar majority classes during cross-domain adaptation.

CADA is built on the DINO detector. On top of standard supervised source-domain training and unlabeled target-domain self-training, it introduces dynamic cross-domain class confusion modeling, confusion-guided prototype contrastive learning, and class-wise fine-grained domain alignment. These components provide more stable and sufficient optimization signals for minority classes during domain adaptation.

## Highlights

- **CRCM: Class-Relation Construction Module**  
  Dynamically maintains a cross-domain class confusion matrix using the prediction distribution of target-domain object queries, explicitly modeling confusion relationships where minority classes drift toward majority classes.

- **CGCL: Confusion-Guided Contrastive Learning**  
  Uses the confusion matrix as a hard negative mining signal and applies stronger separation constraints to easily confused class prototypes, improving prototype discriminability.

- **CFA: Class-wise Fine-grained Alignment**  
  Assigns an independent domain discriminator to each class and performs source-target alignment at the class-prototype level, preventing majority classes from dominating the overall domain alignment process.
