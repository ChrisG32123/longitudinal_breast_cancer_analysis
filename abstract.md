### Abstract:
Breast cancer accounts for nearly 30% of all new cancer diagnoses in women [1], and response to neoadjuvant therapy remains a critical predictor of long-term outcomes [2]. This project investigates the predictive power of longitudinal radiomics by incorporating multi–time point MRI data from the I-SPY2 clinical trial [3], [4]. We analyze dynamic contrast-enhanced (DCE) MRI scans from 384 patients, each imaged at up to four distinct time points during treatment. Shape-based radiomic features—including tumor volume, sphericity, and maximum 3D diameter—are extracted from segmented tumor regions to quantify morphological changes over time. Building on prior studies that used radiomic percent changes or deep learning models to classify pathological complete response (pCR) [5]–[15], we replicate these results for validation and explore novel modeling strategies. Specifically, we introduce the concept of radiomic “feature velocities”—rates of change in feature values relative to temporal spacing between scans—to better characterize treatment response trajectories. By explicitly modeling time intervals within neural network architectures, we aim to improve prediction on irregularly sampled clinical time series. This work provides a foundation for more temporally aware radiomic modeling and offers new insights into optimizing response prediction in breast cancer.

### References:

[1] American Cancer Society, Cancer Facts & Figures 2024.

[2] N. M. Hylton et al., “Locally advanced breast cancer: MR imaging for prediction of response to neoadjuvant chemotherapy—results from ACRIN 6657/I-SPY TRIAL,” Radiology, vol. 263, no. 3, pp. 663–672, 2012.

[3] W. Li et al., “I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2),” The Cancer Imaging Archive, 2022. [Online]. Available: https://doi.org/10.7937/TCIA.D8Z0-9T85

[4] D. C. Newitt et al., “ACRIN 6698/I-SPY2 Breast DWI,” The Cancer Imaging Archive, 2021. [Online]. Available: https://doi.org/10.7937/TCIA.KK02-6D95

[5] S. C. Partridge et al., “Diffusion-weighted MRI findings predict pathologic response in neoadjuvant treatment of breast cancer: The ACRIN 6698 multicenter trial,” Radiology, vol. 289, no. 3, pp. 618–627, 2018.

[6] W. Li et al., “Predicting breast cancer response to neoadjuvant treatment using multi-feature MRI: results from the I-SPY 2 TRIAL,” npj Breast Cancer, vol. 6, no. 63, 2020.

[7] Y. Huang et al., “Longitudinal MRI-based fusion novel model predicts pathological complete response in breast cancer treated with neoadjuvant chemotherapy: a multicenter, retrospective study,” EClinicalMedicine, vol. 58, 2023.

[8] H. Dammu, T. Ren, and T. Q. Duong, “Deep learning prediction of pathological complete response, residual cancer burden, and progression-free survival in breast cancer patients,” PLoS One, vol. 18, no. 1, e0280148, 2023.

[9] E. J. Sutton et al., “Longitudinal breast MRI and radiomic biomarkers for monitoring response to neoadjuvant chemotherapy: A pilot study,” Radiology: Imaging Cancer, vol. 1, no. 4, p. e180026, 2019.

[10] Y. Gao et al., “An explainable longitudinal multi-modal fusion model for predicting neoadjuvant therapy response in women with breast cancer,” Nat. Commun., vol. 15, no. 9613, 2024.

[11] Z. Zhou et al., “Prediction of pathologic complete response to neoadjuvant systemic therapy in triple negative breast cancer using deep learning on multiparametric MRI,” Sci. Rep., vol. 13, p. 1171, 2023.

[12] A. Syed et al., “Machine learning with textural analysis of longitudinal multiparametric MRI and molecular subtypes accurately predicts pathologic complete response in patients with invasive breast cancer,” PLoS One, vol. 18, no. 1, e0280320, 2023.

[13] J. H. Gierach et al., “Relationship of MRI background parenchymal enhancement to breast cancer risk: A literature review and future directions,” J. Magn. Reson. Imaging, vol. 47, no. 1, pp. 33–44, 2018.

[14] Z. Zhou et al., “Prediction of pathologic complete response to neoadjuvant systemic therapy in triple negative breast cancer using deep learning on multiparametric MRI,” Sci. Rep., vol. 13, p. 1171, 2023.

[15] Y. Gao et al., “An explainable longitudinal multi-modal fusion model for predicting neoadjuvant therapy response in women with breast cancer,” Nat. Commun., vol. 15, no. 9613, 2024.

