# NAS_Bench_Reproduce
reproduce some algos in NAS-Bench-201

# run
python ./exp/file.py --config_file configFilePath

# results
| Algorithm      | time（搜索代码运行时间 500） | query | Auto DL (500) | accuracy (avg 500) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Random      |  0.026  | 107.1 | 90.94 / 93.72 | 90.82 / 93.59 |
| Regular EA   |  0.03   | 102.9 | 91.02 / 93.80 | 90.93 / 93.70 |
| Reinforce |  0.125  | 98.9 | 90.20 / 93.12 | 90.27 / 93.19 |
| ENAS | - | - |

1. Reinforce < Random < REA
2. Reinforce的值无论是开源代码还是复现代码都明显低于论文值，原因？ 需要注意一下原文的实验细节
   原论文没有发现确切的lr是多少，但根据附录的lr图，lr < 0.2时整体数值和论文数值贴近。

# Reinforce
| Auto DL (12000s)|  ours (12000s) | paper (12000s) | ours lr 0.1 (12000s) | ours steps 500 |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 90.20 / 93.12  |  90.27 / 93.19  |  | 91.09 / 93.85 | 91.14 / 93.97 | 90.45 / 93.35 |

# REA
| Auto DL (12000s)|  ours sample 3 (12000s) | ours sample 10 (12000s) | paper (12000s) |
| ----------- | ----------- | ----------- | ----------- |
| 91.02 / 93.80  |  90.93 / 93.70  |  | 91.23 / 93.99 | 91.19 / 93.92 |