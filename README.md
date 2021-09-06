# NAS_Bench_Reproduce
reproduce some algos in NAS-Bench-201

# run
python ./exp/file.py --config_file configFilePath

# results
| Algorithm      |  Accuracy(cifar10 avg 10) | time（搜索代码运行时间） | query
| ----------- | ----------- | ----------- |
| Random      |    91.03 /  93.72  |   0.026    | 106.5 |
| Regular EA   |    90.99 / 93.64   |  0.03     | 108.2 |
| Reinforce |  90.19 / 92.81   |  0.125   | 98.9 |
| ENAS | - | - |