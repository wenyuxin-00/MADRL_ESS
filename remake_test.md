# REMAKE 升格执行记录

本文件记录本次迁移后的主线约束。当前仓库以 REMAKE 重构结果作为唯一主项目，不再保留 `REMAKE/` 包、旧项目源码、旧 notebook、旧测试实现或旧 notebook record。

## 当前主线

根目录保留：

```text
configs/
controllers/
data/
datasets/
envs/
models/
notebooks/
predictors/
scripts/
tests/
utils/
artifacts/
README.md
remake_test.md
requirements.txt
pytest.ini
AGENTS.md
CLAUDE.md
```

`configs/cfg.py` 是唯一配置入口。`utils/paths.py` 以仓库根目录作为 `PROJECT_ROOT`，所有运行产物写入 `artifacts/runs/<run_id>/`。

## Notebook 合同

所有 notebook 使用根目录主线 import，不允许 `REMAKE.*` import。根目录定位逻辑寻找：

```python
path / "configs" / "cfg.py"
```

默认复用完整 run：

```python
run_dir = Path("artifacts/runs/20260504_234339_01e73bd1")
```

`madrl_base_safe.ipynb` 默认合同：

```python
scheme_name = "madrl_base_safe"
retrain = False
train_episodes = 500
```

只有明确需要重新训练时，才手动把 `retrain` 改为 `True`。

## Tests 合同

旧项目测试不再作为可执行合同。当前 `tests/` 只保留：

```text
test_artifact_contracts.py
test_encoding_hygiene.py
test_notebook_contracts.py
test_remake_smoke.py
```

测试覆盖：

- UTF-8 和 notebook 文本卫生
- notebook import、根目录定位、默认 run 和 `retrain = False`
- artifacts 完整性
- 小配置端到端 smoke：forecast、share_data、MADRL、eval、compare

## Artifacts 合同

只保留完整主线 run：

```text
artifacts/runs/20260504_234339_01e73bd1
```

旧 run `20260504_234328_01e73bd1` 已删除。完整 run 至少包含 config、share_data manifest、forecast artifacts、7 个 compare record、3 个 MADRL meta 和核心 compare 表。

## 验收命令

```powershell
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/compare.ipynb
gitnexus detect_changes --scope all --repo MADRL_ESS
```

GitNexus 变更风险为 CRITICAL，符合整仓升格预期；需要重点关注 `GridEnv`、eval、MPC、MADRL 和 compare 相关执行流。
