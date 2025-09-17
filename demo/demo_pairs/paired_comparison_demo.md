# Image Pair Comparison Demo: User Disagreements

## Overview
This demo shows image pairs where our top two users disagree on their perception labels, along with baseline and personalized captions for each image.

## Users
- **User 1**: Best performer (87.1% accuracy)
- **User 2**: Second best (84.8% accuracy)

## Label Key
- **1**: First image preferred
- **2**: Second image preferred
- **Bold**: Indicates disagreement between users

---

## Pair 1

<table>
<tr>
<td width="50%" align="center"><img src="pair1_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair1_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a woman with red hair and sunglasses smiling</td><td>a woman holding a baby in her arms</td></tr>
<tr><td><b>User 1</b></td><td>a woman with red hair and sunglasses smiles at the camera</td><td>a woman holding a baby in her arms</td></tr>
<tr><td><b>User 2</b></td><td>a woman with red hair and sunglasses smiles at the camera</td><td>a woman with a baby in her arms</td></tr>
</table>

---

## Pair 2

<table>
<tr>
<td width="50%" align="center"><img src="pair2_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair2_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a woman wearing sunglasses and smiling at the camera</td><td>a woman with long brown hair and blue eyes smiles at the camera</td></tr>
<tr><td><b>User 1</b></td><td>a woman in sunglasses stands next to another woman</td><td>a woman with long brown hair and blue eyes smiles at the camera</td></tr>
<tr><td><b>User 2</b></td><td>a woman wearing sunglasses and smiling for the camera</td><td>a woman with long hair smiling at the camera</td></tr>
</table>

---

## Pair 3

<table>
<tr>
<td width="50%" align="center"><img src="pair3_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair3_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>Trustworthy</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a woman in a suit and tie looking at the camera</td><td>a woman with long blonde hair and blue eyes</td></tr>
<tr><td><b>User 1</b></td><td>a woman with long black hair and a white shirt</td><td>a woman with long blonde hair standing in front of a tree</td></tr>
<tr><td><b>User 2</b></td><td>a woman in a suit and tie looking at the camera</td><td>a woman with long blonde hair and blue eyes</td></tr>
</table>

---

## Pair 4

<table>
<tr>
<td width="50%" align="center"><img src="pair4_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair4_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a man with long black hair and a black jacket</td><td>a man in a white shirt is looking at the camera</td></tr>
<tr><td><b>User 1</b></td><td>a man with long hair sitting on a couch</td><td>a man in a white shirt is looking into the camera</td></tr>
<tr><td><b>User 2</b></td><td>a man with long hair holding a glass of wine</td><td>a man in a white shirt is looking at the camera</td></tr>
</table>

---

## Pair 5

<table>
<tr>
<td width="50%" align="center"><img src="pair5_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair5_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a man with black hair and a white shirt</td><td>a man in a black jacket standing next to the water</td></tr>
<tr><td><b>User 1</b></td><td>a man with black hair and a white shirt</td><td>a man standing in front of a body of water</td></tr>
<tr><td><b>User 2</b></td><td>a man with black hair and a white shirt</td><td>a man in a black jacket standing next to a body of water</td></tr>
</table>

---

## Pair 6

<table>
<tr>
<td width="50%" align="center"><img src="pair6_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair6_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a woman with blonde hair and blue eyes</td><td>a woman with long black hair and earrings</td></tr>
<tr><td><b>User 1</b></td><td>a woman with blonde hair and blue eyes</td><td>a woman with long black hair and a blue shirt</td></tr>
<tr><td><b>User 2</b></td><td>a woman with blonde hair and blue eyes</td><td>a woman with long black hair and large hoop earrings</td></tr>
</table>

---

## Pair 7

<table>
<tr>
<td width="50%" align="center"><img src="pair7_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair7_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a man wearing a hard hat and a red hard hat</td><td>a woman holding a baby in her arms</td></tr>
<tr><td><b>User 1</b></td><td>a man wearing a red hard hat and a green shirt</td><td>a woman holding a baby in her arms</td></tr>
<tr><td><b>User 2</b></td><td>a man wearing a red hard hat with a badge on it</td><td>a woman holding a baby in her arms</td></tr>
</table>

---

## Pair 8

<table>
<tr>
<td width="50%" align="center"><img src="pair8_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair8_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a woman wearing sunglasses and looking at the camera</td><td>a young girl with long brown hair smiles at the camera</td></tr>
<tr><td><b>User 1</b></td><td>a woman wearing sunglasses is looking at the camera</td><td>a young girl with long brown hair smiles at the camera</td></tr>
<tr><td><b>User 2</b></td><td>a woman wearing sunglasses and looking into the camera</td><td>a young girl with long brown hair smiles at the camera</td></tr>
</table>

---

## Pair 9

<table>
<tr>
<td width="50%" align="center"><img src="pair9_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair9_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a woman with long brown hair and blue eyes</td><td>a woman smiling with her eyes closed</td></tr>
<tr><td><b>User 1</b></td><td>a woman with long hair wearing a black top</td><td>a woman smiling with her eyes closed</td></tr>
<tr><td><b>User 2</b></td><td>a woman with long brown hair and blue eyes</td><td>a woman smiling with her eyes closed</td></tr>
</table>

---

## Pair 10

<table>
<tr>
<td width="50%" align="center"><img src="pair10_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair10_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 2</td><td>Image 2</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 2</td><td>Image 2</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a bearded man with glasses and a beard</td><td>two men wearing sunglasses and smiling at the camera</td></tr>
<tr><td><b>User 1</b></td><td>a bearded man with glasses and a beard</td><td>two men wearing sunglasses and smiling at the camera</td></tr>
<tr><td><b>User 2</b></td><td>a man with a beard wearing glasses and a blue jacket</td><td>two men wearing sunglasses and smiling at the camera</td></tr>
</table>

---

## Summary

- **Total Pairs Analyzed**: 10
- **Total Attribute Disagreements**: 17
- **Most Common Disagreement**: trustworthy

The personalized captions reflect each user's visual perception preferences learned from their comparison labels. Notice how the captions may emphasize different aspects of the images based on which image each user preferred for different attributes.
