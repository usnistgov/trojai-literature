# TrojAI Literature Review 

The list below contains curated papers and arXiv articles that are related to Trojan attacks, backdoor attacks, and data poisoning on neural networks and machine learning systems. They are ordered approximately from most to least recent and articles denoted with a "*" mention the TrojAI program directly. Some of the particularly relevant papers include a summary that can be accessed by clicking the "Summary" drop down icon underneath the paper link. These articles were identified using variety of methods including:

- A [flair](https://github.com/flairNLP/flair) embedding created from the arXiv CS subset; details will be provided later.
- A trained [ASReview](https://asreview.readthedocs.io/en/latest/) random forest model
- A curated manual literature review

1. [Backdoor Learning: A Survey](http://arxiv.org/abs/2007.08745)
1. [Backdoor Attacks and Countermeasures on Deep Learning: A Comprehensive Review](http://arxiv.org/abs/2007.10760)
1. [Live Trojan Attacks on Deep Neural Networks](http://arxiv.org/abs/2004.11370)
1. [Odyssey: Creation, Analysis and Detection of Trojan Models](https://arxiv.org/abs/2007.08142)
1. [Data Poisoning Attacks Against Federated Learning Systems](http://arxiv.org/abs/2007.08432)
1. [Blind Backdoors in Deep Learning Models](http://arxiv.org/abs/2005.03823)
1. [Deep Learning Backdoors](http://arxiv.org/abs/2007.08273)
1. [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084)
1. [Backdoor Attacks on Facial Recognition in the Physical World](http://arxiv.org/abs/2006.14580)
1. [Graph Backdoor](http://arxiv.org/abs/2006.11890)
1. [Backdoor Attacks to Graph Neural Networks](http://arxiv.org/abs/2006.11165)
1. [You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion](http://arxiv.org/abs/2007.02220)
1. [Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks](http://arxiv.org/abs/2007.02343)
1. [Trembling triggers: exploring the sensitivity of backdoors in DNN-based face recognition](https://doi.org/10.1186/s13635-020-00104-z)
1. [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks](https://arxiv.org/abs/2006.12557)
1. [Adversarial Machine Learning -- Industry Perspectives](https://arxiv.org/abs/2002.05646)
1. [ConFoc: Content-Focus Protection Against Trojan Attacks on Neural Networks](https://arxiv.org/abs/2007.00711)
1. [Model-Targeted Poisoning Attacks: Provable Convergence and Certified Bounds](https://arxiv.org/abs/2006.16469)
1. [Deep Partition Aggregation: Provable Defense against General Poisoning Attacks](https://arxiv.org/abs/2006.14768)
1. [The TrojAI Software Framework: An OpenSource tool for Embedding Trojans into Deep Learning Models*](https://arxiv.org/abs/2003.07233)
1. [BadNL: Backdoor Attacks Against NLP Models](https://arxiv.org/abs/2006.01043)
    <details>
      <summary>
      Summary
      </summary>  

      * Introduces first example of backdoor attacks against NLP models using Char-level, Word-level, and Sentence-level triggers (these different triggers operate on the level of their descriptor) 
        * Word-level trigger picks a word from the target model’s dictionary and uses it as a trigger
        * Char-level trigger uses insertion, deletion or replacement to modify a single character in a chosen word’s location (with respect to the sentence, for instance, at the start of each sentence) as the trigger.
        * Sentence-level trigger changes the grammar of the sentence and use this as the trigger
      * Authors impose an additional constraint that requires inserted triggers to not change the sentiment of text input
      * Proposed backdoor attack achieves 100% backdoor accuracy with only a drop of 0.18%, 1.26%, and 0.19% in the models utility, for the IMDB, Amazon, and Stanford Sentiment Treebank datasets
    </details>
1. [Neural Network Calculator for Designing Trojan Detectors*](https://arxiv.org/abs/2006.03707)
1. [Dynamic Backdoor Attacks Against Machine Learning Models](https://arxiv.org/abs/2003.03675)
1. [Vulnerabilities of Connectionist AI Applications: Evaluation and Defence](https://arxiv.org/abs/2003.08837)
1. [Backdoor Attacks on Federated Meta-Learning](https://arxiv.org/abs/2006.07026)
1. [Defending Support Vector Machines against Poisoning Attacks: the Hardness and Algorithm](https://arxiv.org/abs/2006.07757)
1. [Backdoors in Neural Models of Source Code](https://arxiv.org/abs/2006.06841)
1. [A new measure for overfitting and its implications for backdooring of deep learning](https://arxiv.org/abs/2006.06721)
1. [An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks](https://arxiv.org/abs/2006.08131)
1. [MetaPoison: Practical General-purpose Clean-label Data Poisoning](https://arxiv.org/abs/2004.00225)
1. [Backdooring and Poisoning Neural Networks with Image-Scaling Attacks](https://arxiv.org/abs/2003.08633)
1. [Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability](https://arxiv.org/abs/2005.00191)
1. [On the Effectiveness of Mitigating Data Poisoning Attacks with Gradient Shaping](https://arxiv.org/abs/2002.11497)
1. [A Survey on Neural Trojans](https://eprint.iacr.org/2020/201.pdf)
1. [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)
    <details>
      <summary>
      Summary
      </summary>  

      * Authors introduce a run-time based trojan detection system called STRIP or STRong Intentional Pertubation which focuses on models in computer vision
      * STRIP works by intentionally perturbing incoming inputs (ie. by image blending) and then measuring entropy to determine whether the model is trojaned or not. Low entropy violates the input-dependance assumption for a clean model and thus indicates corruption 
      * Authors validate STRIPs efficacy on MNIST,CIFAR10, and GTSRB acheiveing false acceptance rates of below 1%
    </details>
1. [TrojDRL: Trojan Attacks on Deep Reinforcement Learning Agents](https://arxiv.org/abs/1903.06638)
1. [Regula Sub-rosa: Latent Backdoor Attacks on Deep Neural Networks](https://arxiv.org/abs/1905.10447)
1. [Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems](https://arxiv.org/abs/1908.03369)
1. [TBT: Targeted Neural Network Attack with Bit Trojan](https://arxiv.org/abs/1909.05193)
1. [Bypassing Backdoor Detection Algorithms in Deep Learning](https://arxiv.org/abs/1905.13409)
1. [A backdoor attack against LSTM-based text classification systems](https://arxiv.org/abs/1905.12457)
1. [Invisible Backdoor Attacks Against Deep Neural Networks](https://arxiv.org/abs/1909.02742)
1. [Detecting AI Trojans Using Meta Neural Analysis](https://arxiv.org/abs/1910.03137)
1. [Label-Consistent Backdoor Attacks](https://arxiv.org/abs/1912.02771)
1. [NeuronInspect: Detecting Backdoors in Neural Networks via Output Explanations](https://arxiv.org/abs/1911.07399)
1. [Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs](https://arxiv.org/abs/1906.10842)
1. [Programmable Neural Network Trojan for Pre-Trained Feature Extractor](https://arxiv.org/abs/1901.07766)
1. [Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection](https://arxiv.org/abs/1908.00686)
1. [TamperNN: Efficient Tampering Detection of Deployed Neural Nets](https://arxiv.org/abs/1903.00317)
1. [TABOR: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems](https://arxiv.org/abs/1908.01763)
1. [Design of intentional backdoors in sequential models](https://arxiv.org/abs/1902.09972)
1. [Design and Evaluation of a Multi-Domain Trojan Detection Method on Deep Neural Networks](https://arxiv.org/abs/1911.10312)
1. [Poison as a Cure: Detecting &amp; Neutralizing Variable-Sized Backdoor Attacks in Deep Neural Networks](https://arxiv.org/abs/1911.08040)
1. [Data Poisoning Attacks on Stochastic Bandits](https://arxiv.org/abs/1905.06494)
1. [Hidden Trigger Backdoor Attacks](https://arxiv.org/abs/1910.00033)
1. [Deep Poisoning Functions: Towards Robust Privacy-safe Image Data Sharing](https://arxiv.org/abs/1912.06895)
1. [A new Backdoor Attack in CNNs by training set corruption without label poisoning](https://arxiv.org/abs/1902.11237)
1. [Deep k-NN Defense against Clean-label Data Poisoning Attacks](https://arxiv.org/abs/1909.13374)
1. [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets](https://arxiv.org/abs/1905.05897)
1. [Revealing Backdoors, Post-Training, in DNN Classifiers via Novel Inference on Optimized Perturbations Inducing Group Misclassification](https://arxiv.org/abs/1908.10498)
1. [Explaining Vulnerabilities to Adversarial Machine Learning through Visual Analytics](https://arxiv.org/abs/1907.07296)
1. [Subpopulation Data Poisoning Attacks](https://www.ccis.northeastern.edu/home/jagielski/subpop_finance.pdf)
1. [TensorClog: An imperceptible poisoning attack on deep neural network applications](https://ieeexplore.ieee.org/document/8668758)
1. [Deepinspect: A black-box trojan detection and mitigation framework for deep neural networks](https://cseweb.ucsd.edu/~jzhao/files/DeepInspect-IJCAI2019.pdf)
1. [Resilience of Pruned Neural Network Against Poisoning Attack](https://ieeexplore.ieee.org/document/8659362)
1. [Spectrum Data Poisoning with Adversarial Deep Learning](https://arxiv.org/abs/1901.09247)
1. [Neural cleanse: Identifying and mitigating backdoor attacks in neural networks](https://people.cs.uchicago.edu/~huiyingli/publication/backdoor-sp19.pdf)
1. [SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems](https://arxiv.org/abs/1812.00292)
    <details>
      <summary>
      Summary
      </summary>  

      * Authors develop SentiNet detection framework for locating universal attacks on neural networks
      * SentiNet is ambivalent to the attack vectors and uses model visualization / object detection techniques to extract potential attacks regions from the models input images.  The potential attacks regions are identified as being the parts that influence the prediction the most. After extraction, SentiNet applies these regions to benign inputs and uses the original model to analyze the output 
      * Authors stress test the SentiNet framework on three different types of attacks— data poisoning attacks, Trojan attacks, and adversarial patches. They are able to show that the framework achieves competitive metrics across all of the attacks  (average true positive rate of 96.22% and an average true negative rate of 95.36%) 
    </details>
1. [PoTrojan: powerful neural-level trojan designs in deep learning models](https://arxiv.org/abs/1802.03043)
1. [Hardware Trojan Attacks on Neural Networks](https://arxiv.org/abs/1806.05768)
1. [Spectral Signatures in Backdoor Attacks](https://arxiv.org/abs/1811.00636)
    <details>
      <summary>
      Summary
      </summary>  

      * Identified a  "spectral signatures" property of current backdoor attacks which allows the authors to use robust statistics to stop Trojan attacks 
      * The "spectral signature" refers to a change in the covariance spectrum of learned feature representations that is left after a network is attacked. This can be detected by using singular value decomposition (SVD). SVD is used to identify which examples to remove from the training set. After these examples are removed the model is retrained on the cleaned dataset and is no longer Trojaned. The authors test this method on the CIFAR 10 image dataset.
    </details>
1. [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/abs/1811.03728)
    <details>
      <summary>
      Summary
      </summary>  

      * Proposes Activation Clustering approach to backdoor detection/ removal which analyzes the neural network activations for anomalies and works for both text and images
      * Activation Clustering uses dimensionality techniques (ICA, PCA) on the activations and then clusters them using k-means (k=2) along with a silhouette score metric to separate poisoned from clean clusters  
      * Shows that Activation Clustering is successful on three different image/datasets (MNIST, LISA, Rotten Tomatoes)  as well as in settings where multiple Trojans are inserted and classes are multi-modal 
    </details>
1. [Model-Reuse Attacks on Deep Learning Systems](https://arxiv.org/abs/1812.00483)
1. [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459)
1. [Trojaning Attack on Neural Networks](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech)
1. [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)
    <details>
      <summary>
      Summary
      </summary>  

      * Proposes neural network poisoning attack that uses "clean labels" which do not require the adversary to mislabel training inputs
      * The paper also presents a optimization based method for generating their poisoning attacks and provides a watermarking strategy for end-to-end attacks that improves the poisoning reliability 
      * Authors demonstrate their method by using generated poisoned frog images from the CIFAR
dataset to manipulate different kinds of image classifiers
    </details>
1. [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185)
    <details>
      <summary>
      Summary
      </summary>  

      * Investigate two potential detection methods for backdoor attacks  (Fine-tuning and pruning). They find both are insufficient on their own and thus propose a combined detection method which they call "Fine-Pruning"  
      * Authors go on to show that on three backdoor techniques "Fine-Pruning" is able to eliminate or reduce Trojans on datasets in the traffic sign, speech, and face recognition domains  
    </details>
1. [Technical Report: When Does Machine Learning FAIL? Generalized Transferability for Evasion and Poisoning Attacks](https://arxiv.org/abs/1803.06975)
1. [Backdoor Embedding in Convolutional Neural Network Models via Invisible Perturbation](https://arxiv.org/abs/1808.10307)
1. [Hu-Fu: Hardware and Software Collaborative Attack Framework against Neural Networks](https://arxiv.org/abs/1805.05098)
1. [Attack Strength vs. Detectability Dilemma in Adversarial Machine Learning](https://arxiv.org/abs/1802.07295)
1. [Data Poisoning Attacks in Contextual Bandits](https://arxiv.org/abs/1808.05760)
1. [BEBP: An Poisoning Method Against Machine Learning Based IDSs](https://arxiv.org/abs/1803.03965)
1. [Generative Poisoning Attack Method Against Neural Networks](https://arxiv.org/abs/1703.01340)
1. [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733)
    <details>
      <summary>
      Summary
      </summary>  

      * Introduce Trojan Attacks— a type of attack where an adversary can create a maliciously trained network (a backdoored neural network, or a BadNet) that has state-of-the-art performance on the user’s training and validation samples, but behaves badly on specific attacker-chosen inputs
      * Demonstrate backdoors in a more realistic scenario by creating a U.S. street sign classifier that identifies stop signs as speed limits when a special sticker is added to the stop sign

    </details>
1. [Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization](https://arxiv.org/abs/1708.08689)
1. [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526)
1. [Neural Trojans](https://arxiv.org/abs/1710.00942)
1. [Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization](https://arxiv.org/abs/1708.08689)
1. [Certified defenses for data poisoning attacks](https://arxiv.org/abs/1706.03691)
1. [Data Poisoning Attacks on Factorization-Based Collaborative Filtering](https://arxiv.org/abs/1608.08182)
1. [Data poisoning attacks against autoregressive models](https://dl.acm.org/doi/10.5555/3016100.3016102)
1. [Using machine teaching to identify optimal training-set attacks on machine learners](https://dl.acm.org/doi/10.5555/2886521.2886721)
1. [Poisoning Attacks against Support Vector Machines](https://arxiv.org/abs/1206.6389)
1. [Antidote: Understanding and defending against poisoning of anomaly detectors](https://dl.acm.org/doi/10.1145/1644893.1644895)
