# Quantum Matrix Multiplication (Swap Test Approach)

This project implements matrix multiplication using a **hybrid quantum-classical algorithm** based on the **swap test**, inspired by:

Schuld et al., *Quantum Algorithms for Matrix Multiplication*  
https://arxiv.org/abs/1803.01601

---

## 🚀 Overview

Matrix multiplication is computed using:

$$
C_{ij} = \text{row}_i(A) \cdot \text{col}_j(B)
$$

Instead of computing dot products classically, this project:

- Encodes vectors as quantum states
- Uses the **swap test** to estimate overlap
- Reconstructs the dot product using:
  - quantum magnitude
  - classical sign
  - norm rescaling

---

## 🧠 Key Idea

For normalized vectors, the swap test gives:

$$
p_0 = \frac{1 + \left|\langle v \mid w \rangle\right|^2}{2}
$$

We recover the magnitude of the inner product:

$$
\left|\langle v \mid w \rangle\right| = \sqrt{2p_0 - 1}
$$

Then compute the full dot product:

$$
v \cdot w =
\text{sign}(v \cdot w)
\cdot \left|\langle v \mid w \rangle\right|
\cdot \|v\| \, \|w\|
$$

---

## ⚙️ Features

- Supports matrix multiplication:

$$
A_{m \times n} \cdot B_{n \times p}
$$

- Works for **non-square matrices**
- Supports **inner dimension ≤ 3**
- Uses:
  - amplitude encoding (2 qubits per vector)
  - swap test circuit (5 qubits total)
- Hybrid reconstruction for real-valued matrices

---

## 📁 Project Structure
