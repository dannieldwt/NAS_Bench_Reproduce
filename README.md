# NAS_Bench_Reproduce
reproduce some algos in NAS-Bench-201

# run
python ./exp/file.py --config_file configFilePath

# results
| Algorithm      | time（搜索代码运行时间 500） | query | Auto DL (500) | accuracy (avg 500) |
| ----------- | ----------- | ----------- |
| Random      |  0.026  | 107.1 | 90.94 / 93.72 | 90.82 / 93.59 |
| Regular EA   |  0.03   | 102.9 | 91.02 / 93.80 | 90.93 / 93.70 |
| Reinforce |  0.125  | 98.9 | 90.20 / 93.12 | 90.27 / 93.19 |
| ENAS | - | - |

1. Reinforce < Random < REA
2. Reinforce的值无论是开源代码还是复现代码都明显低于论文值，原因？ 需要注意一下原文的实验细节
3. query过低？query应当是多少？
4. 训练停止的时间真的是12000s吗？  是的，但是是bench里的时间 就是包括了训练和验证的时间耗费