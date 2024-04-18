# FLAMINGO: Adaptive and Resilient Federated Meta-Learning against Adversarial Attacks

## Abstract

In today’s data-centric world, the interplay between
Meta Learning and Federated Learning (FL) heralds a new era of
technological advancement, driving rapid adaptation, improved
model generalization, and collaborative model training across
decentralized networks. This fusion, known as Federated Meta-
Learning (FML), emerges as a cutting-edge solution for resource-
restricted edge devices, enabling the production of personalized
models with limited training data. However, FML navigates a
complex terrain, balancing efficiency with security, particularly
as adversarial attacks on edge devices pose significant threats.
These attacks risk introducing bias and undermining the integrity
of model training, a critical concern given the typically sparse
data on edge devices. This paper explores the intricate dynamics
of FML amidst such adversarial challenges, introducing a novel
algorithm, FLAMINGO. FLAMINGO is designed to conduct
adversarial meta-training coupling with data augmentation and
consistency regularization strategies, thereby strengthening the
meta-learner’s defenses against adversarial attacks. This strategic
approach not only protects meta-learners against adversarial
threats but also prevents overfitting, striving a balance between
privacy, security, and technological efficiency, all while optimizing
communication costs in the FML landscape

## Training with Attack

In FML_attack.sh supply paths for the datasets. Model requires root directory of the EuroSAT and LISA dataset. Lisa Dataset has to be in image format. Then run in your bash terminal

```
./bash/FML_attack.sh
```

## Adversarial Federated Meta Training

To train the model with adversarial federated meta-training, you need to provide the path of your dataset. Modify the `adv_train.sh` script by replacing `"/path/to/dataset"` with the actual path to your dataset. Then run

```
./bash/adv_train.sh
```
