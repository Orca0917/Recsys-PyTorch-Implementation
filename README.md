<div align="center">
<h1> Recommender System PyTorch Implementation </h1>
</div>

<br/>

## 👋 Introduction

Hello! The repository you are currently viewing is personally studying recommended system models
using deep learning and storing code implemented using PyTorch. You can use it freely at any time,
but if there is something wrong or there is a problem with the execution environment, 
please let me know through issue.

<br/>

## 📑 Paper lists

I show you a list of papers that I organized and implemented myself.
Please note that we have referred to other implementation codes for some models,
and you can check the original code referenced in the folder.
The contents of the thesis are being organized on the blog and are being organized in Korean.
For more information, visit my blog as well!


| Index | Year |  Model  |                                                                              Paper Link                                                                               |                   Blog Link                    |              Implementation              |
|:-----:|:----:|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------:|:----------------------------------------:|
|   1   | 2017 |  NeuMF  |  [Paper](https://dl.acm.org/doi/abs/10.1145/3038912.3052569?casa_token=0Mn-nBbA8DkAAAAA:1GLqj8Yb63TLSKBwCvl9NNzpQLOWO7mgBto24pPGMd9rlDU9Mic5fZm73VcxBZy6tCxzWN_odg)   | [Blog](https://killerwhale0917.tistory.com/33) |             [Code](./NeuMF)              |
|   2   | 2010 |   FM    | [Paper](https://ieeexplore.ieee.org/abstract/document/5694074?casa_token=y8NcEPGtCNkAAAAA:TwepAyJVyImsVxxx6N-AmT-V5auhy9mdegF2bN9LkSiStis3k01Kc_EEdHPLvp8CTS1AA1nfEg) | [Blog](https://killerwhale0917.tistory.com/40) | [Code](./Factorization%20Machine%20(FM)) |
|   3   | 2015 | AutoRec |                                                   [Paper](https://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf)                                                   |                   [Blog](#)                    |            [Code](./AutoRec)             |


<br/>

## 🛠️ How to run the code?

Differences may exist in how each model is run.
Therefore, you can refer to the README written in each model's folder and run it.
Basically, it is unified by running train.py , but in some cases it is not, so **please read README**.
For smooth execution, please download and set the `requirements.txt` package attached to each folder.

```shell
pip install -r requirements.txt
python3 train.py
```
