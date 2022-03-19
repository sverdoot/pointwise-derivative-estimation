# Advanced Statistical Methods, final project

- [Advanced Statistical Methods, final project](#advanced-statistical-methods-final-project)
  - [Getting started](#getting-started)
  - [Overview](#overview)
  

## Getting started

```bash
conda create --name asm  python==3.9
```

if poetry is not installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

```bash
conda activate asm
```

```bash
python asm/main.py --configs configs/project.yml
```

## Overview

Unbiased risk estimate is computed as:

$\hat{\mathcal{R}}(\hat{f}) = \|\mathcal{K}_{h} \mathbf{Y} - \mathbf{Y}\|^2 + 2\sigma^2\text{tr}(\mathcal{K}_h)$

True risk is computed as:

$\mathcal{R}(\hat{f}) = \mathbb{E}((f^*)'(x_0) - \hat{f}'(x_0))^2$