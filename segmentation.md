## I-SPY 2 DCE “FTV Analysis Mask” Bit Encoding (Revised)

---

## 1. Overview

Each voxel in the I-SPY 2 “FTV Analysis Mask” is stored as a small integer whose bits indicate why it was excluded (or included) during the functional tumor volume (FTV) pipeline.  Specifically:  
- **A final value of `1` (binary `000001₂`) marks an included tumor voxel**—i.e., it survived all filters and lies inside the reader’s bounding box.   
- **A final value of `17` (binary `010001₂`) marks a background voxel inside the bounding box**—i.e., non-tumor tissue that lies inside the VOI.   
- **Other values** combine bits to indicate which exclusion steps applied (e.g., PE threshold, “omit” region, outside VOI, or outside FOV).  

> **Key point:** The I-SPY 2 pipeline relabels “passed all filters → included tumor” voxels to **`1`** (instead of raw 0), and labels “background_inside_box” voxels as **`17`** (instead of raw 16) so that a simple test `mask == 1` isolates tumor .

---

## 2. Bit Positions and Their Meanings

The mask uses six bit positions (0 through 5).  When a given bit _k_ (0–5) is set to 1, it indicates the voxel was flagged (excluded or labeled) at that step.  Below is the corrected mapping:

| **Bit Index (k)** | **Decimal Value** | **Meaning When Bit k = 1**                                                                                                                                                          | **Citation**                                  |
|:-----------------:|:-----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|
| **0**             | `1`               | **Included tumor (bright) inside the bounding box**: Voxel survived all exclusion filters and is final tumor ROI.                                                                    |                            |
| **1**             | `2`               | **Failed minimum percent-enhancement (PE) threshold**: Voxel did not reach the required early post-contrast enhancement (e.g., < 70 % PE).                                           |                            |
| **2**             | `4`               | **Failed 3D connectivity check**: Voxel was not connected to the main enhancing cluster (isolated speckle).                                                                            |                            |
| **3**             | `8`               | **Inside a manual “OMIT” region**: Voxel lies within a reader-defined polygon (e.g., vessel, benign region) that should be excluded.                                                  |                            |
| **4**             | `16`              | **Background (non-tumor) inside the bounding box**: Voxel was manually determined to be non-tumor tissue within the VOI.                                                             |                            |
| **5**             | `32`              | **Extra tissue (outside the bounding box or cropped FOV)**: Voxel lies outside the reader’s VOI or outside the cropped breast field of view (e.g., chest wall, padding, contralateral breast). |                            |

> **Note:** In pure bit-flag logic, “bit 0 = 1” would normally mean “excluded at background threshold.”  However, in I-SPY 2’s *final* mask, the pipeline flips that interpretation:  
> - **`1`** (= bit 0 only) → included tumor inside VOI (not “excluded”)  
> - **`17`** (= bits 4 + 0) → background inside VOI (not “included”)   

---

## 3. Decoding Common Decimal Values

Below are six commonly seen mask values.  For each, we show its 6-bit binary form (bit 5…bit 0), which bits are set, and what that means in plain language.

### 3.1 Mask = `1` (`000001₂`)

- **Binary (bits 5…0):**
- **Bits set:** 0  
- **Meaning:**  
- **bit 0 = 1** → “Included tumor (bright) inside the bounding box.”  
- **bits 1–4 = 0** → passed PE threshold, connectivity check, omit-region test, and is inside VOI.  
- **bit 5 = 0** → inside cropped FOV.  
- **Plain language:**  
> **Mask = 1** ⇒ Voxel is tumor (bright) and lies inside the reader’s bounding box.   

---

### 3.2 Mask = `17` (`010001₂` or `10001₂` for bits 0–4)

- **Binary (bits 5…0):**
- **Bits set:** 4 + 0  
- **Pipeline override:**  
- By I-SPY 2 convention, **`17`** does **not** mean “outside VOI + background”; instead, it is reserved to label “background inside VOI” (non-tumor). 
- **bit 4 = 1** → this bit alone marks “background inside VOI.”  
- **bit 0 = 1** in raw logic means “bright,” but because bit 4 has priority in I-SPY 2’s final labeling, **any** “background_inside_box” voxel is set to 17.  
- **Plain language:**  
> **Mask = 17** ⇒ Voxel is background (non-tumor) within the bounding box (VOI).   

---

### 3.3 Mask = `32` (`100000₂`)

- **Binary (bits 5…0):**
- **Bits set:** 5  
- **Meaning:**  
- **bit 5 = 1** → “Extra tissue outside the bounding box / cropped FOV” (e.g., chest wall, padding, contralateral breast).  
- **bits 0–4 = 0** → would have passed all internal tumor filters if it were inside VOI.  
- **Plain language:**  
> **Mask = 32** ⇒ Voxel lies outside the bounding box (extra tissue, not considered in analysis).   

---

### 3.4 Mask = `33` (`100001₂`)

- **Binary (bits 5…0):**
- **Bits set:** 5 + 0  
- **Meaning:**  
- **bit 5 = 1** → “Outside VOI / extra tissue.”  
- **bit 0 = 1** in raw logic means “bright.”  
- Because bit 5 dominates, **Mask = 33** corresponds to a bright voxel outside the bounding box (e.g., bright chest wall).   
- **Plain language:**  
> **Mask = 33** ⇒ A bright (enhancing) voxel that lies outside the bounding box (extra tissue such as chest wall). 

---

### 3.5 Mask = `34` (`100010₂`)

- **Binary (bits 5…0):**  
- **Bits set:** 5 + 1  
- **Meaning:**  
- **bit 5 = 1** → “Outside VOI / extra tissue.”  
- **bit 1 = 1** → “Failed PE threshold” (i.e., < 70 % percent‐enhancement).  
- **bits 0, 2–4 = 0** → would have passed background, connectivity, omit, VOI filters if inside VOI.  
- **Plain language:**  
> **Mask = 34** ⇒ Voxel lies outside the bounding box (extra tissue) and also failed the percent‐enhancement threshold. 

---

### 3.6 Mask = `49` (`110001₂`)

- **Binary (bits 5…0):**  
- **Bits set:** 5 + 4 + 0  
- **Meaning:**  
- **bit 5 = 1** → “Outside VOI / extra tissue.”  
- **bit 4 = 1** → “Background inside VOI,” but because this voxel is already flagged bit 5, it is effectively “background outside VOI.”  
- **bit 0 = 1** → In raw logic “bright,” but that is overridden by bit 4 and bit 5.  
- **Plain language:**  
> **Mask = 49** ⇒ Voxel is background (non-tumor) and lies outside the bounding box (extra tissue).

---

## 4. Quick Reference Table

| **Decimal** | **Binary (bits 5…0)** | **Interpretation**                                                                  |
|:-----------:|:---------------------:|:------------------------------------------------------------------------------------|
| **1**       | `000001₂`              | Tumor voxel (bright) inside VOI.                                     |
| **17**      | `010001₂`              | Background (non-tumor) inside VOI.                                |
| **32**      | `100000₂`              | Extra tissue (outside VOI).                                       |
| **33**      | `100001₂`              | Bright voxel outside VOI (e.g. chest wall).                    |
| **34**      | `100010₂`              | Extra tissue + failed PE threshold.                              |
| **49**      | `110001₂`              | Background + outside VOI.                                  |

> **Bits (5→0)** correspond to:  
> - **bit 5 (32)** = extra tissue outside VOI/FOV  
> - **bit 4 (16)** = background inside VOI  
> - **bit 3 (8)**  = manual “omit” region (unused here)  
> - **bit 2 (4)**  = connectivity check fail (unused here)  
> - **bit 1 (2)**  = failed percent‐enhancement threshold
> - **bit 0 (1)**  = included tumor inside VOI 

---

## 5. Extracting a Binary Mask


### Tumor Mask
To isolate just the true tumor voxels (mask = 1), use:

```python
tumor_mask = (mask_arr == 1).astype(np.uint8)
```

### Bounding Box / Volume Of Interest Mask
To isolate the entire bounding box voxels, we include the true tumor voxels (mask = 1) and non-tumorous voxels inside the bounding box (mask = 17), use:

```python
bb_mask = ((mask_arr == 1) | (mask_arr == 17)).astype(np.uint8)
```

## References

1. **I-SPY 2 Collection on TCIA**  
   The Cancer Imaging Archive (TCIA). “I-SPY2 Breast MRI Collection.”  
   https://www.cancerimagingarchive.net/collection/ispy2/   

2. **Analysis Mask Files Description**  
   I-SPY 2 Investigators. “Analysis mask files description (v20211020).”  
   https://www.cancerimagingarchive.net/wp-content/uploads/Analysis-mask-files-description.v20211020.docx   

3. **ACRIN-6698 ISPY2 DWI and DCE MRI Data Descriptions**  
   I-SPY 2 Investigators. “ACRIN-6698 ISPY2 DWI and DCE MRI Data Descriptions (20210520).”  
   https://www.cancerimagingarchive.net/wp-content/uploads/ACRIN-6698-ISPY2-DWI-and-DCE-MRI-Data-Descriptions_20210520.pdf   

4. **Predicting Breast Cancer Response (npj Breast Cancer)**  
   Li, W., Newitt, D. C., Gibbs, J., et al. “Predicting breast cancer response to neoadjuvant treatment using multi-feature MRI: results from the I-SPY 2 TRIAL.” *npj Breast Cancer* 6, 16 (2020).  
   https://doi.org/10.1038/s41523-020-00203-7   

5. **Large-Scale DCE-MRI Benchmark (arXiv)**  
   [Anonymous]. “A large-scale multicenter breast cancer DCE-MRI benchmark for automated segmentation and radiomic feature extraction.” *arXiv:2406.13844v3* (2024).  
   https://arxiv.org/abs/2406.13844v3   

6. **Predicting Breast Cancer Response (ResearchGate)**  
   [Anonymous]. “Predicting breast cancer response to neoadjuvant treatment using multi-feature MRI: results from the I-SPY 2 TRIAL.” *ResearchGate.*  
   https://www.researchgate.net/publication/347200154_Predicting_breast_cancer_response_to_neoadjuvant_treatment_using_multi-feature_MRI_results_from_the_I-SPY_2_TRIAL   

7. **Biomarkers Consortium – I-SPY TRIAL-2**  
   Foundation for the National Institutes of Health (FNIH). “Biomarkers Consortium – I-SPY TRIAL-2: Investigation of Serial Studies To Predict Your Therapeutic Response With Imaging and Molecular Analysis…”  
   https://fnih.org/our-programs/biomarkers-consortium-i-spy-trial-2-investigation-of-serial-studies-to-predict-your-therapeutic-response-with-imaging-and-molecular-analysis-an-adaptive-breast-cancer-trial-design-in-the-setting/   

8. **Breast MRI 2.0 – ICPME**  
   [Anonymous]. “I-SPY 2: an adaptive breast cancer trial design in the setting of neoadjuvant chemotherapy.” *ICPME.*  
   https://www.icpme.us/courses/breast/SBMR_Breast%20MRI.pdf   

9. **Tumor Morphology for Prediction of Poor Responses Early In I-SPY 2 (PMC)**  
   [Anonymous]. “Tumor Morphology for Prediction of Poor Responses Early in I-SPY 2.” *PMC Article.*  
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11598075/   

10. **Molecular Hallmarks of Breast Multiparametric MRI (PMC)**  
    I-SPY 2 Investigators. “Molecular hallmarks of breast multiparametric magnetic resonance imaging (mpMRI) phenotypes.” *PMC Article.*  
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9860227/ 