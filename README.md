# 问题二优化版代码

这套代码用于解决：

1. 目标区域 100% 时间单重覆盖；
2. 在满足覆盖的方案中尽量减少总卫星数；
3. 计算 95% 时间全区域二重覆盖指标；
4. 输出成本和覆盖缺口图。

## 文件说明

- `problem2_model.py`：统一轨道、覆盖和评价模型，不要随意改。
- `search_problem2.py`：三阶段搜索 M、N、i、F。
- `validate_problem2.py`：对一个候选方案做详细验证和画图。
- `requirements.txt`：依赖库。

## VS Code 中运行

在该文件夹打开终端：

```bash
python -m pip install -r requirements.txt
python search_problem2.py
```

搜索完成后，打开 `results/stage3_fine_results.csv`，找到 `single_feasible=True` 且总卫星数最少的方案。

把该方案填入 `validate_problem2.py` 顶部的 `CANDIDATE`，然后运行：

```bash
python validate_problem2.py
```

## 验证层级

搜索阶段先使用较粗网格，最后必须提高精度。建议：

- 初步验证：1°、60 s、24 h；
- 论文最终验证：0.5°、30 s、72 h。

最终论文中应写：

> 在 0.5° 空间步长、30 s 时间步长和 72 h 仿真范围内未检测到覆盖空隙。

不要把有限离散仿真表述成数学意义上的绝对证明。

## 当前可先验证的候选

`validate_problem2.py` 默认填入：

- M = 38
- N = 44
- i = 53°
- F = 1
- 总卫星数 = 1672

该方案在 1°、60 s、24 h 的当前模型测试中通过了单重覆盖；但仍需用更密网格和更长时间进行最终验证。
