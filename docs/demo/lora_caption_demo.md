# LoRA Personalized Image Captioning Demo

## Overview
This demo shows how LoRA-adapted CLIP models generate personalized captions based on individual user preferences learned from the FFHQ face perception dataset.

## Model Details
- **Base Model**: CLIP ViT-B/32
- **Caption Model**: BLIP (Salesforce/blip-image-captioning-base)
- **LoRA Configuration**: Rank 4, Alpha 1.0
- **Training**: Visual prompts with user-specific tokens
- **Dataset**: FFHQ face images with human perception labels

## Results

The following images show:
1. **Baseline Caption**: Standard CLIP-BLIP caption without personalization
2. **User 1 Caption**: Best performing user's personalized caption
3. **User 2 Caption**: Second best user's personalized caption

---

### Image 1: 00241.webp

![00241.jpg](00241.jpg)

**Baseline Caption:**  
`two women are smiling and posing for the camera`

**User 1 (Best, 87.1%) Caption:**  
`two women smiling and posing for the camera`  
*Embedding shift: 1.123*

**User 2 (2nd, 84.8%) Caption:**  
`two women smiling and posing for the camera`  
*Embedding shift: 1.017*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 2: 10653.webp

![10653.jpg](10653.jpg)

**Baseline Caption:**  
`a photo of a woman in a headscar`

**User 1 (Best, 87.1%) Caption:**  
`an image of a woman in a headscar`  
*Embedding shift: 1.082*

**User 2 (2nd, 84.8%) Caption:**  
`a woman in a headscar is shown in this photo`  
*Embedding shift: 1.101*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 3: 06347.webp

![06347.jpg](06347.jpg)

**Baseline Caption:**  
`a man and a woman holding up an american flag`

**User 1 (Best, 87.1%) Caption:**  
`a man and a woman holding up an american flag`  
*Embedding shift: 1.248*

**User 2 (2nd, 84.8%) Caption:**  
`two people posing for the camera with a red, white and blue flower in their hand`  
*Embedding shift: 1.054*

**Changes detected:** User 2 (2nd, 84.8%)

---

### Image 4: 29694.webp

![29694.jpg](29694.jpg)

**Baseline Caption:**  
`a woman and a young boy smile for the camera`

**User 1 (Best, 87.1%) Caption:**  
`a woman and a young boy smile for the camera`  
*Embedding shift: 1.068*

**User 2 (2nd, 84.8%) Caption:**  
`a woman and a young boy smiling for the camera`  
*Embedding shift: 0.979*

**Changes detected:** User 2 (2nd, 84.8%)

---

### Image 5: 29989.webp

![29989.jpg](29989.jpg)

**Baseline Caption:**  
`a young boy in a red jacket looking at the camera`

**User 1 (Best, 87.1%) Caption:**  
`a little boy in a red jacket looking at the camera`  
*Embedding shift: 0.998*

**User 2 (2nd, 84.8%) Caption:**  
`a young boy in a red jacket looking at the camera`  
*Embedding shift: 1.041*

**Changes detected:** User 1 (Best, 87.1%)

---

### Image 6: 15729.webp

![15729.jpg](15729.jpg)

**Baseline Caption:**  
`a woman with purple hair and horns on her head`

**User 1 (Best, 87.1%) Caption:**  
`a woman with purple hair and horns holding a cell`  
*Embedding shift: 1.231*

**User 2 (2nd, 84.8%) Caption:**  
`a woman with purple hair and horns on her head`  
*Embedding shift: 1.089*

**Changes detected:** User 1 (Best, 87.1%)

---

### Image 7: 07870.webp

![07870.jpg](07870.jpg)

**Baseline Caption:**  
`a woman with long blonde hair and blue eyes`

**User 1 (Best, 87.1%) Caption:**  
`a woman with long blonde hair standing in front of a microphone`  
*Embedding shift: 1.216*

**User 2 (2nd, 84.8%) Caption:**  
`a woman with long blonde hair standing in front of a microphone`  
*Embedding shift: 1.267*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 8: 04560.webp

![04560.jpg](04560.jpg)

**Baseline Caption:**  
`a little girl wearing a colorful necklace`

**User 1 (Best, 87.1%) Caption:**  
`a little girl wearing a colorful beaded necklace`  
*Embedding shift: 1.153*

**User 2 (2nd, 84.8%) Caption:**  
`a young boy wearing a colorful beaded necklace`  
*Embedding shift: 1.179*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 9: 08694.webp

![08694.jpg](08694.jpg)

**Baseline Caption:**  
`a woman wearing sunglasses and holding a book`

**User 1 (Best, 87.1%) Caption:**  
`a woman wearing sunglasses and holding a book`  
*Embedding shift: 1.370*

**User 2 (2nd, 84.8%) Caption:**  
`a woman wearing sunglasses and holding a book`  
*Embedding shift: 1.351*

**Note:** Captions unchanged (subtle embedding shifts only)

---

### Image 10: 10931.webp

![10931.jpg](10931.jpg)

**Baseline Caption:**  
`a woman with long black hair and a white shirt`

**User 1 (Best, 87.1%) Caption:**  
`a woman with long black hair looking at the camera`  
*Embedding shift: 1.188*

**User 2 (2nd, 84.8%) Caption:**  
`a woman with long black hair is looking at the camera`  
*Embedding shift: 1.141*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 11: 18531.webp

![18531.jpg](18531.jpg)

**Baseline Caption:**  
`a woman with long black hair smiles at the camera`

**User 1 (Best, 87.1%) Caption:**  
`a woman with long hair smiles at the camera`  
*Embedding shift: 0.866*

**User 2 (2nd, 84.8%) Caption:**  
`a woman smiling and looking at the camera`  
*Embedding shift: 0.837*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 12: 14508.webp

![14508.jpg](14508.jpg)

**Baseline Caption:**  
`a man and woman are smiling for the camera`

**User 1 (Best, 87.1%) Caption:**  
`a man and woman posing for the camera`  
*Embedding shift: 1.055*

**User 2 (2nd, 84.8%) Caption:**  
`a man and woman are smiling for the camera`  
*Embedding shift: 0.933*

**Changes detected:** User 1 (Best, 87.1%)

---

### Image 13: 18222.webp

![18222.jpg](18222.jpg)

**Baseline Caption:**  
`an asian woman with long brown hair and a black jacket`

**User 1 (Best, 87.1%) Caption:**  
`a woman with long brown hair and a black jacket`  
*Embedding shift: 1.125*

**User 2 (2nd, 84.8%) Caption:**  
`a woman with brown hair and a black jacket`  
*Embedding shift: 1.028*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 14: 15890.webp

![15890.jpg](15890.jpg)

**Baseline Caption:**  
`a young boy with blonde hair smiling at the camera`

**User 1 (Best, 87.1%) Caption:**  
`a young boy with blonde hair smiling at the camera`  
*Embedding shift: 1.402*

**User 2 (2nd, 84.8%) Caption:**  
`a young boy in a black shirt smiles at the camera`  
*Embedding shift: 1.291*

**Changes detected:** User 2 (2nd, 84.8%)

---

### Image 15: 14383.webp

![14383.jpg](14383.jpg)

**Baseline Caption:**  
`a man with glasses and a white shirt`

**User 1 (Best, 87.1%) Caption:**  
`a young man wearing glasses and smiling at the camera`  
*Embedding shift: 1.028*

**User 2 (2nd, 84.8%) Caption:**  
`a young man wearing glasses and smiling at the camera`  
*Embedding shift: 1.070*

**Changes detected:** User 1 (Best, 87.1%), User 2 (2nd, 84.8%)

---

### Image 16: 08487.webp

![08487.jpg](08487.jpg)

**Baseline Caption:**  
`a young man with blonde hair smiles at the camera`

**User 1 (Best, 87.1%) Caption:**  
`a young man with blonde hair smiles at the camera`  
*Embedding shift: 0.952*

**User 2 (2nd, 84.8%) Caption:**  
`a young man with blonde hair smiles at the camera`  
*Embedding shift: 1.021*

**Note:** Captions unchanged (subtle embedding shifts only)

---

### Image 17: 18761.webp

![18761.jpg](18761.jpg)

**Baseline Caption:**  
`a woman with orange paint on her face`

**User 1 (Best, 87.1%) Caption:**  
`a woman with orange paint on her face`  
*Embedding shift: 1.109*

**User 2 (2nd, 84.8%) Caption:**  
`a woman with orange paint on her face`  
*Embedding shift: 1.122*

**Note:** Captions unchanged (subtle embedding shifts only)

---

### Image 18: 05431.webp

![05431.jpg](05431.jpg)

**Baseline Caption:**  
`a man is standing in front of a car`

**User 1 (Best, 87.1%) Caption:**  
`a man standing in the middle of a road`  
*Embedding shift: 1.058*

**User 2 (2nd, 84.8%) Caption:**  
`a man is standing in front of a car`  
*Embedding shift: 1.143*

**Changes detected:** User 1 (Best, 87.1%)

---

### Image 19: 00273.webp

![00273.jpg](00273.jpg)

**Baseline Caption:**  
`a woman wearing sunglasses and smiling at the camera`

**User 1 (Best, 87.1%) Caption:**  
`a woman wearing sunglasses and smiling at the camera`  
*Embedding shift: 1.324*

**User 2 (2nd, 84.8%) Caption:**  
`a woman wearing sunglasses and smiling at the camera`  
*Embedding shift: 1.293*

**Note:** Captions unchanged (subtle embedding shifts only)

---

### Image 20: 18852.webp

![18852.jpg](18852.jpg)

**Baseline Caption:**  
`a young boy with his tongue sticking out`

**User 1 (Best, 87.1%) Caption:**  
`a young boy with his tongue sticking out`  
*Embedding shift: 1.223*

**User 2 (2nd, 84.8%) Caption:**  
`a young boy with his tongue sticking out`  
*Embedding shift: 1.163*

**Note:** Captions unchanged (subtle embedding shifts only)

---

## Summary Statistics

- **Total Images Processed**: 20
- **User 1 (Best, 87.1%) Caption Changes**: 12/20 (60%)
- **User 2 (2nd, 84.8%) Caption Changes**: 11/20 (55%)

### Average Embedding Shifts
- **User 1 (Best, 87.1%)**: 1.141
- **User 2 (2nd, 84.8%)**: 1.106


## Key Observations

The personalized captions demonstrate how individual users' visual preferences (learned from their perception labels) influence image description:

1. **Subtle Changes**: Some captions show minor word choice differences
2. **Attribute Focus**: Different users may emphasize different visual attributes
3. **Emotional Context**: Some users' models add emotional or contextual details
4. **Consistent Shifts**: All images show significant embedding shifts even when captions don't change

## Technical Notes

- Visual prompt tokens are inserted after the CLS token in the ViT architecture
- Each user has a learned 768-dimensional visual prompt token
- LoRA adapters modify the MLP layers in the visual transformer
- Caption generation uses different sampling strategies to reflect personalization
