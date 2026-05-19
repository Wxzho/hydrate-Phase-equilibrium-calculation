# Gas Hydrate Phase-Equilibrium Hybrid Model

气体水合物相平衡 **vdW-P 物理模型 + 机器学习残差修正** 混合预测框架。  
核心代码：`hydrate_model.py`（无绘图依赖，适合仓库发布）。

## 功能概览

- **物理层**：Peng–Robinson 状态方程、Kihara 势、van der Waals–Platteeuw (vdW-P) 多组分平衡压力
- **结构选择**：纯组分 / 同结构混合物 / sI–sII 热力学竞争规则
- **ML 层**：LightGBM + Random Forest 集成，在对数残差空间修正物理预测
- **热力学输出**：平衡压力、生成焓 ΔH、笼穴占有率 θ、抑制剂水活度
- **训练流程**：数据质量过滤、分组交叉验证 (LOSO)、零假设基线对比、模型持久化

## 环境要求

- Python **3.9+**
- 见 [`requirements.txt`](requirements.txt)

## 安装

```bash
git clone <your-repo-url>
cd <repo-directory>
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 项目文件

| 文件 | 说明 |
|------|------|
| `hydrate_model.py` | 主程序（训练 / 预测） |
| `date.csv` | 训练数据集（需自行提供或放入仓库） |
| `hydrate_parameters.xlsx` | 组分物性、Kihara、BIP 参数（首次训练可自动创建模板） |
| `hybrid_*.joblib` | 训练后生成的模型与配置（`predict` 模式需要） |

### 训练数据格式 (`date.csv`)

必需列示例：

| 列名 | 含义 |
|------|------|
| `T(K)` | 温度 (K) |
| `p(MPa)` | 实验平衡压力 (MPa) |
| `xCH4`, `xC2H6`, `xC3H8`, `xCO2`, `xN2`, `xH2S`, `xi-C4H10` | 各组分摩尔分数 |

可选列：`source` / `reference` 等，用于 Leave-One-Source-Out 交叉验证。

## 使用方法

### 1. 训练模型

编辑 `hydrate_model.py` 底部：

```python
if __name__ == '__main__':
    RUN_MODE = 'train'   # 或 'both'
```

在项目目录下执行：

```bash
python hydrate_model.py
```

训练完成后将生成（默认前缀 `hybrid`）：

- `hybrid_lgbm_model.joblib`
- `hybrid_rf_model.joblib`
- `hybrid_scaler_X.joblib`
- `hybrid_model_config.joblib`
- `test_set_predictions_detailed_corrected.csv`
- `dataset_composition_summary.csv`
- `supplementary_model_files/`（可复现性补充导出）

若不存在 `hydrate_parameters.xlsx`，程序会自动调用 `create_parameter_template()` 创建参数模板。

### 2. 新工况预测

训练完成后，将 `RUN_MODE` 设为 `'predict'` 或 `'both'`：

```python
predict_pressure_for_new_conditions(
    temperature=277.15,
    composition_dict={'Methane': 1.0},
)
```

或在其他脚本中导入：

```python
from hydrate_model import predict_pressure_for_new_conditions

result = predict_pressure_for_new_conditions(
    temperature=280.0,
    composition_dict={
        'Methane': 0.85,
        'Carbon dioxide': 0.15,
    },
    inhibitor_type='methanol',          # 可选
    inhibitor_weight_fraction=0.10,   # 可选
)

print(result['pressure_hybrid'])      # MPa
print(result['delta_H_kJ_mol'])       # kJ/mol
print(result['theta_small_cage'])     # 小笼占有率
```

### 3. 仅使用物理模型（无需 ML 权重）

```python
from hydrate_model import predict_structure_and_pressure

p, structure, p_sI, p_sII = predict_structure_and_pressure(
    temperature=277.15,
    components=['Methane'],
    composition=[1.0],
)
```

## 组分名称

`composition_dict` / `components` 请使用下列标准名称之一：

`Methane`, `Ethane`, `Propane`, `Carbon dioxide`, `Nitrogen`, `Hydrogen Sulfide`, `i-Butane`

## 预测方程

\[
P_{\mathrm{hybrid}} = \exp\bigl(\ln P_{\mathrm{vdW\text{-}P}} + 0.65\, r_{\mathrm{LGB}} + 0.35\, r_{\mathrm{RF}}\bigr)
\]

其中 \(r\) 为训练目标 \(\ln P_{\mathrm{exp}} - \ln P_{\mathrm{vdW\text{-}P}}\) 的 ML 预测值。

## 目录说明

- `hydrate_model.py` — 发布用精简版（无 matplotlib / 绘图）
- `hydrate_model_core.py` — 完整版（含论文级绘图与详细注释，可选，一般不随仓库发布）

## 许可证

请在发布前补充适用的开源许可证（如 MIT）。若数据 `date.csv` 来自文献，请遵守原始数据的引用与使用条款。

## 引用

若本代码用于学术论文，请引用相应手稿及 vdW-P / PR-EOS 原始文献。
