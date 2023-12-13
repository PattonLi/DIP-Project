# MystMarkdown Usage

## build bool

```bash
ghp-import -n -p -f _build/html
jupyter-book build ./
```

(my-label)=
## use of myst markdown

```{note}
:class: dropdown
Here is a note sssss
```

```{warning}
warning
```

`````{admonition} This admonition was styled...
:class: tip
With a tip class!
`````

---
分割线

````{tab-set}
```{tab-item} Tab 1 title
My first tab
```

```{tab-item} Tab 2 title
My second tab with `some code`!
```
````

[导航](my-label)

```{figure} ../images/logo.png
:width: 200px
:align: center
:name: my-fig-ref

My figure title.
```

<img src="../images/logo.png" alt="fishy" width="200px">

```{sidebar} My sidebar title
My sidebar content
```

````{div} full-width
```{note}
Here's a note that will take the full width
```
````

## MATH

$$
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
$$

```{math}
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

## Citations

You can also cite references that are stored in a `bibtex` file. For example,

{cite}`holdgraf_evidence_2014`

```{bibliography}
```

缩放点积注意力是一种注意力机制，其中点积按 $\sqrt{d_k}$ 比例缩小.

如果我们有 a query $Q$, a key $K$ and a value $V$ ，计算attention的公式为:

$$
\text{Attention}(Q,K,V)=\text{softmax}\biggl(\frac{QK^T}{\sqrt{d_k}}\biggr)V
$$

之后我们假设 $q$ 和 $k$ 是 $d_k$-dimensional 向量其分量是均值为0，方差为1的独立随机变量，那么它们的点积为$q\cdot k=\sum_{i=1}^{dk}u_iv_i$ 均值为 0 同时方差为
 $d_k$.
 
因为我们希望这些值的方差为1, 所以将计算结果除以 $\sqrt{d_k}$