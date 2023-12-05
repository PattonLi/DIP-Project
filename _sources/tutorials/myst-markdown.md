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

## ssss

sdasda


<select onchange="changeImage(event)">
  <option value="image1">图片1</option>
  <option value="image2">图片2</option>
  <option value="image3">图片3</option>
</select>

<img id="imageToShow" src="">

<script>
function changeImage(event) {
  var selectedValue = event.target.value;
  var imageToChange = document.getElementById("imageToShow");
  
  if (selectedValue === "image1") {
    imageToChange.src = "../images/model-result/A-100-1.png";
  } else if (selectedValue === "image2") {
    imageToChange.src = "../images/model-result/B-100-1.png";
  } else if (selectedValue === "image3") {
    imageToChange.src = "../images/model-result/C-100-1.png";
  }
}
</script>