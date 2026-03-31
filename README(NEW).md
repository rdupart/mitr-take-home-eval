To see results, see results folder.

Main script file is mechanistic_mitr_experiment.py

Answers below:


The current problem is that the base experiments (detailed in results.md and roberta_bert_results.md) only measure accuracy and contradiction rate at the start, and end, but never verify that the internal layers actually became functionally diverse. The output vectors of adjacent layers could look statistically different (satisfying the penalty) while attention heads inside those layers still detect the same patterns. The notebooks (mitr_distilbert_boolq.ipynb and mitr_bert_roberta_boolq.ipynb) apply the penalty to layer output hidden states only, leaving the residual stream and internal attention computations unpenalized. A reviewer would reject this because it claims that MITR improves logical behavior, but there is no probing on each layer to show evidence that the layers are actually doing distinct work. The accuracy improvement could come from generic regularization, not the redundancy mechanism specifically. I can fix this by training the three model BERT, DistilBERT, and RoBERTa with and without MITR, freeze the networks, and run probing classifiers on every layer. If MITR is working as claimed, the probing accuracy profile across layers should shift. The second weakness is that the experiment only penalizes neighboring layers L and L+1, but CKA literature shows in BERT sized models, middle layers cluster regardless of adjacency. A reviewer would know this, and might argue that the researchers aren't targeting where redundancy actually lives in the network. If we aren't targeting where redundancy occurs, then the negative BERT results are explained by the wrong intervention target, rather than a limitation argued by the authors. To fix this, I would compare both the adjacent and global conditions to see what happens in layers close to each other vs ones farther apart. 
MITR, with my additions, asks a mechanistic question: does penalizing representational similarity between adjacent transformer layers force each layer to learn functionally distinct computations, and does this improve logical behavior? This fits within the workshop's core concern — understanding what model internals are actually doing, and whether training interventions reshape that computation in interpretable ways. The probing classifier analysis directly examines how MITR redistributes where logical properties are encoded across layers, using CKA similarity matrices to visualize representational diversity before and after training. The central finding — that only RoBERTa shows a genuine layer-level functional shift while smaller models are unaffected — is a mechanistic result. It tells us something specific about how model expressivity determines whether training-time regularization can rewire internal computation, which is precisely the kind of causal, internals-focused finding this workshop exists to surface.
Two related papers are: "Iteration Head: A Mechanistic Study of Chain-of-Thought" (MI Workshop Orals and Poster Sessions)accepted at ICML 2024 mech-interp workshop; the paper examines how specific transformer components implement multi-step reasoning, relevant to the layer diversification and its relationships with logical consistency. The second paper accepted by the same workshop "Benchmarking Mental State Representations in Language Models" uses probing classifiers across model sizes and training conditions to study how internal representations encode reasoning-relevant properties. This is relevant to our probing experiment, which uses the same technique. 


Name
Layer-wise Probing Analysis
Hypothesis
If we train linear probing classifiers on each frozen layer's representations before and after MITR training, then models where MITR improves logical behavior (RoBERTa) will show shifted probing accuracy profiles across layers, because functional diversification should redistribute where logical properties are encoded.
Expected Result
Probing accuracy for logical properties should shift — some layers gaining and others losing — between baseline and MITR variants. 
What negative results mean
If probing profiles are flat then the accuracy and contradiction improvements are not caused by genuine layer diversification so the theoretical motivation is wrong.
Estimated GPU hours
1 with H200 SXM runpod; (i used up my collab credits)
How this changes the paper's central claim
Changes it from "MITR reduces redundancy and improves logic" to either a confirmed mechanistic claim (if probes shift) or a more honest "MITR regularizes effectively on expressive models for reasons not yet fully understood."





Name
Global vs Adjacent Layer Penalty Comparison
Hypothesis
If we penalize all layer pairs rather than only adjacent pairs, then CKA similarity across the full network will decrease more uniformly and logical behavior metrics will improve more consistently, because CKA analysis shows middle layers cluster together regardless of adjacency in BERT-sized models.
Expected Result
Probing accuracy for logical properties should shift — some layers gaining and others losing — between baseline and MITR variants. 
What negative results mean
If global penalization hurts accuracy or contradiction despite reducing CKA, it means forcing all layers apart is too restrictive — the model needs some redundancy to function, and only local redundancy is harmful.
Estimated GPU hours
1 with H200 SXM runpod; (i used up my collab credits)


How this changes the paper's central claim
Reframes the central claim from "penalizing adjacent redundancy helps" to "the location of the redundancy penalty matters and must match where redundancy actually lives in the network."




Experiment 1 is more important to run first because it attacks the original experiment's claim directly. If probing shows layers are not actually diversifying, the entire theoretical motivation collapses regardless of what the accuracy numbers say. Experiment 2 is an improvement to the method, but it builds on the assumption that the mechanism is real, verified by experiment 1. 
Abstract: Transformer models rely on depth to perform multi-step reasoning, but adjacent layers frequently learn redundant representations, wasting model capacity. We introduce MITR (Mutual Information Transformer Regularization), which penalizes representational similarity between consecutive transformer layers during training to force each layer to contribute distinct information. We evaluate MITR across DistilBERT, BERT, and RoBERTa on BoolQ using four surrogate objectives — CLUB, InfoNCE, cosine similarity, and CKA — measuring both accuracy and logical contradiction rate. Our central finding is that MITR's effectiveness is conditioned on model expressivity: only RoBERTa showed meaningful improvement in contradiction rate (reducing from 0.614 to 0.610) without a significant drop in accuracy (0.761 to 0.758), alongside a genuine shift in layer-wise profiles, with non-adjacent CKA similarity dropping from 0.232 to 0.176 under global penalization. BERT and DistilBERT showed negligible or negative structural effects (CKA remained flat at 0.288 for BERT and 0.245 for DistilBERT), suggesting that forcing layer separation is too restrictive for smaller models with limited representational capacity. These results indicate that redundancy regularization improves logical consistency only when the model has sufficient depth and expressivity to redistribute computation across layers.



