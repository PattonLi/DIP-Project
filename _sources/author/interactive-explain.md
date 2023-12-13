# 交互式说明

>本来这部分是想做交互式的，但是有一个问题就是切换图片显示的时候需要调用函数，此处需要用到jupyter kernel，而转成html之后无法调用python函数。

>并且不太方便手动嵌入html和js代码，这是由于使用的框架的原因。

```{warning}
我们尝试了很多方法，并在stackoverflow上找到了这样的回答：

Note that ipywidgets tend to behave differently from other interactive visualization libraries. They interact both with Javascript, and with Python. Some functionality in ipywidgets may not work in default Jupyter Book pages (because no Python kernel is running). You may be able to get around this with tools for remote kernels, like thebe.
```

