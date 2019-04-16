![](https://img.shields.io/badge/license-MIT-green.svg)
![](https://img.shields.io/badge/PAF-v1.0.0-orange.svg)
![](https://img.shields.io/badge/language-python-yellow.svg)
![](https://img.shields.io/badge/backbone-PyTorch-important.svg)   

## Welcome to PAF - PyTorch Alchemy Furnace（PyTorch炼丹炉）~  

🍳This project aims at providing a rich function environment for Machine Learning with PyTorch.🍳    

🍻🍻**Contributions welcome!** 

In this framework, the `Dataset`, `Net`, `Action` are separated totally. They are powered by `args`.  

The basic framework structure illustrates as below.  

![Framework](https://ws1.sinaimg.cn/mw690/005O8ntygy1g24jfadlh4j333f26o1bu.jpg)  

**Dataset**:  You can define all your datasets here, and use `get_dataloader` function to get its PyTorch dataloader.  

**Net**: This module includes all the models your project needed. And to reuse some sub-models, you can define them in `net_parts`.   

**Action**: The `Action` module occupies a very important position as you can code any action you want to `record` your network results, such as `model graph`,`accuracy`, `confusion matrix`, `auc` and `loss` etc..  

**Agent**:  It is the `spokesman` of the above three modules, because `model` train/eval with `dataset` and using `action` to `record` the performance. So, in this part, you can assign `dataloader`, `model` and `action` by `get_dataloader`, `get_action` and `get_net` function respectively. 

**Config**: This part mainly used to config some directories which can be dirs to store results or dirs referring to datasets. It contains the environment which remains unchanged. It can be useful when you debug locally and deploy remotely.  

**Tool**: It is a `utils` combination mainly used to generate parser. But you can put any `mess` code here.  

**Main**: It is the core of the framework as it provides `args` to other modules. We use `args` to control the whole work.  

**Logger**: This tool is a colorful progressbar logger. It can be used simply as a logger. During training, you can activate its `progressbar` function, so you can view the progress of your program.   

Some screenshot shows as below:  

![](https://ws1.sinaimg.cn/mw690/005O8ntygy1g24pumc1cyj31ng15gguo.jpg)  

![](https://ws1.sinaimg.cn/mw690/005O8ntygy1g24puw1fzwj327i0vkqa3.jpg)  

![](https://ws1.sinaimg.cn/mw690/005O8ntygy1g24pw2a02yj31040tetmw.jpg)
