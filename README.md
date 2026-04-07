# Quantum Matrix Multiplication (Swap Test Approach)

This project implements matrix multiplication using a **hybrid quantum-classical algorithm** based on the **swap test**, inspired by:

> Schuld et al., *Quantum Algorithms for Matrix Multiplication* (arXiv:1803.01601)

The implementation uses:
- Qiskit
- AWS Braket local simulator (`braket_sv`)
- Hybrid classical + quantum workflow

---

## 🚀 Overview

Matrix multiplication is computed using:

\[
C_{ij} = \text{row}_i(A) \cdot \text{col}_j(B)
\]

Instead of computing dot products classically, this project:
1. Encodes vectors as quantum states
2. Uses the **swap test** to estimate overlap
3. Reconstructs the dot product using:
   - quantum magnitude
   - classical sign
   - norm rescaling

---

## 🧠 Key Idea

For normalized vectors:

\[
p_0 = \frac{1 + |\langle v | w \rangle|^2}{2}
\]

We recover:

\[
|\langle v | w \rangle| = \sqrt{2p_0 - 1}
\]

Then compute:

\[
v \cdot w = \text{sign}(v \cdot w) \cdot |\langle v | w \rangle| \cdot ||v|| \cdot ||w||
\]

---

## ⚙️ Features

- Supports matrix multiplication:
  - \( A_{m \times n} \cdot B_{n \times p} \)
- Works for **non-square matrices**
- Supports **inner dimension ≤ 3**
- Uses:
  - amplitude encoding (2 qubits per vector)
  - swap test circuit (5 qubits total)
- Hybrid reconstruction for real-valued matrices

---

## 📁 Project Structure
