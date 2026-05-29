# References

One paper per core technique in OhMyWord's stack.

**Federated Learning for keyboards**
McMahan et al., Google, 2018 — https://arxiv.org/abs/1811.03604
> Architecture OhMyWord's baseline is built on: CIFG-LSTM, 1.4MB, FedAvg.
> Detailed enough to use as a technical spec.

**On-device keyboard model (Apple iOS 17)**
Jack Cook, 2023 — https://jackcook.com/2023/09/08/predictive-text.html
> Reverse-engineered Apple's keyboard: GPT-2 style, 34M params, Core ML.
> Our architecture ceiling reference.

**LoRA: personal adapter layer**
Hu et al., Microsoft, 2021 — https://arxiv.org/abs/2106.09685
> Defines the technique behind OhMyWord's per-user personalization:
> freeze base model, train only small rank-decomposition matrices (~1-2MB per user).

**Knowledge distillation**
Hinton et al., Google, 2015 — https://arxiv.org/abs/1503.02531
> How we train the student from the teacher (Qwen2.5-1.5B → GPT-2 Tiny):
> soft labels + KL divergence instead of hard labels only.
