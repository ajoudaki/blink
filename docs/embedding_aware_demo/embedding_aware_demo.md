# Embedding-Aware Caption Generation Demo

## Overview
This demo generates captions that directly reflect the differences in CLIP embeddings between baseline and user-personalized models.

## Method
1. Calculate attribute similarity scores using CLIP text encoder
2. Compare how these scores change with personalized embeddings
3. Generate captions that emphasize the attributes that changed most

## Users
- **User 1**: 87.1% accuracy
- **User 2**: 84.8% accuracy

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

### Embedding-Aware Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a person</td><td>a person</td></tr>
<tr><td><b>User 1</b></td><td>a person in the photo</td><td>an older person with distinctive features</td></tr>
<tr><td><b>User 2</b></td><td>an older mysterious person with distinctive features</td><td>an older person in the photo</td></tr>
</table>

### Attribute Score Changes (vs baseline)

**Image 1:**
- Attractive: User1=-0.04, User2=-0.14
- Smart: User1=-0.05, User2=-0.20
- Trustworthy: User1=-0.05, User2=-0.16

**Image 2:**
- Attractive: User1=-0.09, User2=-0.06
- Smart: User1=-0.13, User2=-0.09
- Trustworthy: User1=-0.12, User2=-0.08

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

### Embedding-Aware Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a person</td><td>a person</td></tr>
<tr><td><b>User 1</b></td><td>an older person with distinctive features</td><td>an older person with distinctive features</td></tr>
<tr><td><b>User 2</b></td><td>an older person with distinctive features</td><td>an older person with distinctive features</td></tr>
</table>

### Attribute Score Changes (vs baseline)

**Image 1:**
- Attractive: User1=-0.15, User2=-0.14
- Smart: User1=-0.18, User2=-0.16
- Trustworthy: User1=-0.14, User2=-0.13

**Image 2:**
- Attractive: User1=-0.10, User2=-0.08
- Smart: User1=-0.15, User2=-0.14
- Trustworthy: User1=-0.15, User2=-0.12

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

### Embedding-Aware Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a person</td><td>a person</td></tr>
<tr><td><b>User 1</b></td><td>an older person with distinctive features</td><td>an older mysterious person in the photo</td></tr>
<tr><td><b>User 2</b></td><td>an older person with distinctive features</td><td>an older person in the photo</td></tr>
</table>

### Attribute Score Changes (vs baseline)

**Image 1:**
- Attractive: User1=-0.12, User2=-0.12
- Smart: User1=-0.17, User2=-0.17
- Trustworthy: User1=-0.13, User2=-0.14

**Image 2:**
- Attractive: User1=-0.15, User2=-0.12
- Smart: User1=-0.18, User2=-0.16
- Trustworthy: User1=-0.16, User2=-0.13

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

### Embedding-Aware Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a person</td><td>a person</td></tr>
<tr><td><b>User 1</b></td><td>an older person in the photo</td><td>a person in the photo</td></tr>
<tr><td><b>User 2</b></td><td>an older mysterious person with distinctive features</td><td>an older person in the photo</td></tr>
</table>

### Attribute Score Changes (vs baseline)

**Image 1:**
- Attractive: User1=-0.15, User2=-0.17
- Smart: User1=-0.14, User2=-0.20
- Trustworthy: User1=-0.13, User2=-0.18

**Image 2:**
- Attractive: User1=-0.04, User2=-0.09
- Smart: User1=-0.04, User2=-0.10
- Trustworthy: User1=-0.07, User2=-0.10

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

### Embedding-Aware Captions

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a person</td><td>a person</td></tr>
<tr><td><b>User 1</b></td><td>a person in the photo</td><td>an older mysterious person with distinctive features</td></tr>
<tr><td><b>User 2</b></td><td>an older mysterious person in the photo</td><td>an older mysterious person with distinctive features</td></tr>
</table>

### Attribute Score Changes (vs baseline)

**Image 1:**
- Attractive: User1=-0.14, User2=-0.18
- Smart: User1=-0.13, User2=-0.19
- Trustworthy: User1=-0.09, User2=-0.15

**Image 2:**
- Attractive: User1=-0.15, User2=-0.16
- Smart: User1=-0.16, User2=-0.20
- Trustworthy: User1=-0.15, User2=-0.18

---

## Summary

This approach generates captions that:
1. Directly reflect the CLIP embedding differences
2. Emphasize attributes that changed most for each user
3. Show personalized perception through caption variation

The captions now vary based on how each user's model perceives different attributes in the images.
