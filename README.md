# Hoyer Sparseness — Measurement and Projection

A Python implementation of Hoyer's sparseness measure and a sparseness projection operator for vectors and matrices.

Originally developed as part of an undergraduate thesis on **Non-negative Matrix Factorization (NMF) for topic extraction from Indonesian online news**, at the Department of Mathematics, Universitas Indonesia (2015).

---

## Background

Hoyer's sparseness measure quantifies how sparse a non-negative vector is, based on the ratio between its L1 and L2 norms:

$$\text{sparseness}(z) = \frac{\sqrt{n} - \|z\|_1 / \|z\|_2}{\sqrt{n} - 1}$$

where $n$ is the dimension of the vector $z$. The value ranges from 0 (fully dense, all elements equal) to 1 (maximally sparse, only one non-zero element).

In the context of this thesis, Hoyer sparseness was used to control the degree of sparseness in NMF factor matrices (W and H), and to investigate how different sparseness levels (empirically observed to be most interpretable in the range **0.6–0.7**) affect the quality and interpretability of extracted topics.

---

## What This Code Does

This repository implements two core operations:

### 1. Sparseness Measurement
- `Vecsparse(z)` — computes Hoyer sparseness of a vector
- `MatrixSparse(A)` — computes mean Hoyer sparseness across all rows of a matrix

### 2. Sparseness Projection
Given a vector (or matrix), projects it to a **target sparseness value** while preserving its L2 norm — based on the projection algorithm described in Hoyer (2004).

- `VecOperatorsp(x, newsparse, tol)` — projects a vector to a target sparseness level
- `MatrixOperatorSp(A, newsp, tol)` — applies row-wise projection across a matrix
- `operatorsparse(...)` and `Zerochanger(...)` — internal helper functions for the projection algorithm

---

## Usage

```python
import numpy as np
from hoyersparseness import Vecsparse, MatrixSparse, VecOperatorsp, MatrixOperatorSp

# Measure sparseness of a vector
z = np.array([0.8, 0.1, 0.05, 0.05])
print(Vecsparse(z))  # returns a value between 0 and 1

# Measure mean sparseness of a matrix
A = np.random.rand(5, 10)
print(MatrixSparse(A))

# Project a vector to target sparseness 0.65 with tolerance 0.01
new_vec, achieved_sparseness = VecOperatorsp(z, newsparse=0.65, tol=0.01)

# Project all rows of a matrix to target sparseness
new_A, mean_sparseness = MatrixOperatorSp(A, newsp=0.65, tol=0.01)
```

---

## Requirements

```
numpy
scipy (for sparse matrix support via .toarray())
```

Install with:
```
pip install numpy scipy
```

---

## Notes

- Input vectors must be **non-negative** — Hoyer sparseness is defined for non-negative values
- The projection operator iterates until the target sparseness, L1, and L2 constraints are satisfied within the given tolerance
- The `tol` parameter controls convergence — smaller values give more precise results but may require more iterations

---

## Reference

Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness constraints. *Journal of Machine Learning Research*, 5, 1457–1469.
