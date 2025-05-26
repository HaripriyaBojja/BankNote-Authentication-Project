About the Project:

The need for a reliable system that would differentiate genuine banknotes from counterfeit ones using advanced machine learning techniques birthed a "Banknote Authentication System”. The method uses a dataset of four critical attributes that have been derived from the image representation of banknotes: variance, skewness, curtosis, and entropy. These features are processed and fed into XGBoost. XGBoost is chosen as the best model due to its superior performance in terms of accuracy, speed, and scalability. To make it accessible and practical, the deployed web application for the solution shall be done with the help of streamlit upon a trained model. This user-friendly platform allows users to input the variance, skewness, curtosis, and entropy values and give real-time predictions of whether or not a given banknote is genuine or a counterfeit. Thus, the whole paper demonstrates this synergy of cutting-edge machine learning techniques and practical and user-centric design in providing an answer to a problem of critical interest in financial security systems. It significantly plays a role in eliminating risks associated with using counterfeit currency through accurate and real time counterfeit detection capabilities.

Dataset Origin:

Banknotes were scanned using an industrial-grade scanner to capture their fine details. Both authentic and forged banknotes were used for the study. The scanned images were analysed using wavelet transformation to extract statistical features. Wavelet transformation is a technique that decomposes a signal into components of different scales, making it easier to analyse complex structure like textures in images. For authenticating banknotes, the data is primarily extracted from datasets found in sources such as UCI Machine Learning Repository or Kaggle. It includes both genuine and counterfeit banknotes, represented by numerical features extracted from images of the banknotes.

Features (Independent Variables):

•	Variance: It provides the variation of pixel intensity.

•	Skewness: This represents the degree of asymmetry of the distribution of pixels.

•	Curtosis: This quantifies the pixel distribution as to how sharp it is or whether it is flat.

•	Entropy: It calculates the complexity or randomness of the image.

Target (Dependent Variable): 

•	Class: A binary variable indicating the type of banknote: 

▪ 0: Genuine banknote. 

▪ 1: Counterfeit banknote.


Dataset Link:

https://archive.ics.uci.edu/dataset/267/banknote+authentication
