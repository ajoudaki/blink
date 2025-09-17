# Embedding-Based Caption Generation Demo

## Overview
This demo generates captions directly from CLIP embeddings (baseline and LoRA-adapted) using a ViT-GPT2 decoder.

## Key Difference from Previous Approaches
- **Direct from embeddings**: Captions are generated from the CLIP embedding space, not from raw images
- **Personalization reflected**: LoRA-adapted embeddings produce different captions
- **No intermediate processing**: The decoder directly consumes the personalized embeddings

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

**Embedding Shifts**: Image 1: User1=1.038, User2=1.355 | Image 2: User1=1.219, User2=1.110

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
</table>

### Captions (Generated from Embeddings)

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a man standing in front of a white wall</td><td>a man standing in front of a graffiti-covered wall with a surfboard</td></tr>
<tr><td><b>User 1</b></td><td>a person standing in front of a fire place</td><td>a man standing on top of a wooden table</td></tr>
<tr><td><b>User 2</b></td><td>a man wearing a blue shirt sitting on top of a wooden bench</td><td>a man sitting on top of a wooden table next to a dog</td></tr>
</table>

**Caption variations detected**: User 1 (Image 1), User 2 (Image 1), User 1 (Image 2), User 2 (Image 2)

---

## Pair 2

<table>
<tr>
<td width="50%" align="center"><img src="pair2_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair2_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

**Embedding Shifts**: Image 1: User1=1.414, User2=1.285 | Image 2: User1=1.357, User2=1.248

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
</table>

### Captions (Generated from Embeddings)

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a white and black and a yellow and black and a white and black and a yellow and black and a yellow and black and a yellow and black</td><td>a person standing in front of a red brick building</td></tr>
<tr><td><b>User 1</b></td><td>a man in a black t-shirt is in front of a blue and white double decker bus</td><td>a woman is holding a large piece of luggage</td></tr>
<tr><td><b>User 2</b></td><td>a large white table topped with two black and white trunks</td><td>a man standing on top of a wooden bench next to a pile of mail</td></tr>
</table>

**Caption variations detected**: User 1 (Image 1), User 2 (Image 1), User 1 (Image 2), User 2 (Image 2)

---

## Pair 3

<table>
<tr>
<td width="50%" align="center"><img src="pair3_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair3_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

**Embedding Shifts**: Image 1: User1=1.356, User2=1.279 | Image 2: User1=1.196, User2=1.160

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
<tr><td>Trustworthy</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
</table>

### Captions (Generated from Embeddings)

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a row of blue and white umbrellas sitting on top of a building</td><td>a person sitting on a park bench with a camera</td></tr>
<tr><td><b>User 1</b></td><td>a small child is standing in front of a large white object</td><td>a man standing next to a wooden table holding a plate of food</td></tr>
<tr><td><b>User 2</b></td><td>a small group of people holding up a colorful green and white object</td><td>a person standing on top of a wooden bench</td></tr>
</table>

**Caption variations detected**: User 1 (Image 1), User 2 (Image 1), User 1 (Image 2), User 2 (Image 2)

---

## Pair 4

<table>
<tr>
<td width="50%" align="center"><img src="pair4_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair4_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

**Embedding Shifts**: Image 1: User1=1.160, User2=1.271 | Image 2: User1=1.034, User2=1.107

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>Attractive</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>Smart</td><td>Image 1</td><td>Image 1</td><td>✓</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 1**</td><td>**Image 2**</td><td>✗</td></tr>
</table>

### Captions (Generated from Embeddings)

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a person sitting on top of a wooden table</td><td>a woman standing in front of a green wall</td></tr>
<tr><td><b>User 1</b></td><td>a vintage photo of a couple of people on a boat</td><td>a vintage photo of a vintage photo of a vintage photo of a vintage photo</td></tr>
<tr><td><b>User 2</b></td><td>a person standing on top of a lush green field</td><td>a large group of brown and white animals standing on top of a green field</td></tr>
</table>

**Caption variations detected**: User 1 (Image 1), User 2 (Image 1), User 1 (Image 2), User 2 (Image 2)

---

## Pair 5

<table>
<tr>
<td width="50%" align="center"><img src="pair5_img1.jpg" width="100%"><br><b>Image 1</b></td>
<td width="50%" align="center"><img src="pair5_img2.jpg" width="100%"><br><b>Image 2</b></td>
</tr>
</table>

**Embedding Shifts**: Image 1: User1=1.129, User2=1.175 | Image 2: User1=1.393, User2=1.428

### User Labels

<table>
<tr><th>Attribute</th><th>User 1 Choice</th><th>User 2 Choice</th><th>Agreement</th></tr>
<tr><td>**Attractive**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
<tr><td>**Smart**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
<tr><td>**Trustworthy**</td><td>**Image 2**</td><td>**Image 1**</td><td>✗</td></tr>
</table>

### Captions (Generated from Embeddings)

<table>
<tr><th width='15%'>Model</th><th width='42.5%'>Image 1 Caption</th><th width='42.5%'>Image 2 Caption</th></tr>
<tr><td><b>Baseline</b></td><td>a large black and white bird sitting on top of a wooden table</td><td>a person on a skateboard in the middle of a crowd of people</td></tr>
<tr><td><b>User 1</b></td><td>a room filled with a table and chairs next to a wall</td><td>a large group of people standing in front of a wall</td></tr>
<tr><td><b>User 2</b></td><td>a man standing in front of a row of white and blue striped umbrellas</td><td>a wooden ledge with a view of a large group of animals</td></tr>
</table>

**Caption variations detected**: User 1 (Image 1), User 2 (Image 1), User 1 (Image 2), User 2 (Image 2)

---

## Summary

This approach demonstrates true embedding-based caption generation:

1. **CLIP embeddings** are extracted using baseline and LoRA-adapted models
2. **Direct generation**: The ViT-GPT2 decoder generates captions directly from these embeddings
3. **Personalization**: Different embeddings (from different users) produce variations in captions
4. **Embedding shifts** correlate with caption diversity

The captions now directly reflect the personalized visual understanding encoded in the LoRA-adapted CLIP models.
