
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

```{figure} ./images/logo.png
:width: 200px
:align: center
:name: my-fig-ref

My figure title.
```

<img src="./images/logo.png" alt="fishy" width="200px">

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

## ssss

sdasda
