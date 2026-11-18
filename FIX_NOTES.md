# 错误丢失与汇总中断修复 — 设计说明与复跑记录

## 1. 根因（修复前）

1. **预处理错误在 `where` 模式下被整体丢弃。**
   `DataFrameSchemaBackend.validate` 先用一个临时 `ErrorHandler` 跑核心预处理器
   （`add_missing_columns` / `strict_filter_columns` / `set_defaults` /
   `coerce_dtype`），随后因为 schema 含 `where` 检查而**新建并替换**了带
   `ConditionalValidationContext` 的 handler。预处理阶段收集到的 strict / ordered /
   转换错误全部留在旧 handler 里，最终判定只看新 handler，于是：
   - `strict=True` 多一列直接返回原表；
   - `ordered=True` 列顺序错误返回成功；
   - 转换失败只剩后续 `WRONG_DATATYPE`，`DATATYPE_COERCION` 与坏值消失。

2. **Column parser 的非法行布局直接抛 `SchemaDefinitionError`。**
   `_run_column_parsers` / `assert_parser_preserves_rows` 立即抛出，穿透整个 lazy
   汇总，其它列的独立检查没有机会运行；parser 抛普通异常时同样直接逃逸。

3. **parser 输出缺少失败隔离。** 失败 parser 的非法输出可能覆盖有效表，依赖它的
   后续 parser / 检查继续消费非法结果。

4. **非行错误能否 `drop_invalid_rows` 靠字符串启发式判断**
   （`isinstance(failure_cases, str)`），无法覆盖转换/parser 失败，且标记无法跨
   嵌套的内层 `ErrorHandler` 传到汇总端。

## 2. 设计：错误由谁持有、如何跨边界传递

### 2.1 单一 handler + 提前创建的 conditional context

- `ConditionalValidationContext` 在**任何预处理之前**创建，与顶层 `ErrorHandler`
  同生命周期；预处理结束后用 `bind_full_table()` 把物理行表重绑到预处理后的表
  （仅一次，清空旧 mask 缓存；此时还没有任何 `where` 被求值）。
- 顶层 handler 不再被替换，预处理、schema parser、column parser、各组件检查的错误
  都进入同一个 handler，最终由它决定抛 `SchemaErrors` 还是 drop。

### 2.2 非行错误标记随错误对象本身流动

- `SchemaError` 新增 `non_row_error: bool`；`ErrorHandler.mark_non_row_error(err)`
  在标记 handler 的同时给错误对象打标。
- 错误即使被内层 handler 收集、包成 `SchemaErrors` 再抛到外层重新收集，标记仍然
  保留，外层 `collect_error` 会同步置位自身的 `has_non_row_error`。
- 结构错误（strict/ordered/缺列/重名列/错 dtype）、转换未完成
  （`DATATYPE_COERCION`、`WRONG_DATATYPE`）、parser 执行错误（`CHECK_ERROR`）
  都属于非行错误，`drop_invalid_rows` 一律改为抛 `SchemaErrors`。

### 2.3 parser 失败的分支隔离（继续检查的范围）

- **Schema 级 parser**：抛错 → 构造/复用 `SchemaError`（`CHECK_ERROR`，保留
  `original_exc`），收集后**停止 parser 链**，后续步骤对“最后一个有效表”继续，
  非法输出永不替换有效表。
- **Column 级 parser**（`_run_column_parsers_safely`）：
  - 每个 parser 只执行一次；抛普通异常或 `SchemaError`、返回非法行布局
    （行数/索引序列不符）或返回非 Series/DataFrame，都收为 `CHECK_ERROR`，
    定位到实际列名与 parser，原异常存 `original_exc`（隐式链也可回溯）。
  - 失败列：丢弃非法输出、保留有效表，**跳过该列自身的后续 parser 与检查**，
    不编造成功/失败记录。
  - 其它列的 parser 与独立检查照常运行。
  - `regex=True` 按 `get_regex_columns` 展开为实际列逐个处理；一列失败不豁免
    其它匹配列，报告归到实际列名，各自保留原坏值。
- 合法 parser 行为不变：schema parser 可重排/筛行，之后位置按处理后的完整表计算；
  合法列 parser 的改值照常生效；parser/where/check_fn 保持原有一次执行粒度，
  check-only 副本保证 parser 不会为跑检查而二次执行。

### 2.4 内存与状态隔离

- 仍然只在 `ErrorHandler.collect_error` 时把 `SchemaError.data` 置 `None`，
  诊断只保留必要的失败值/行归属（窄 canonical frame），不为每条错误复制宽表；
  `SchemaErrors.data` 只持有一份共享表。
- 所有状态都在每次 `validate` 新建的 handler/context 上，不挂到 schema 或
  `Check`/`Parser`；连续、交错、失败后的再次调用互不串扰，schema 组件配置不被改坏，
  `inplace=False` 不改调用方输入。

## 3. 人工小表：期望 vs 实际（可复跑）

脚本：`~/work/repro/expectations_table.py`

| # | 场景 | 期望错误种类 / 坏值 / 回调 | 实际 |
|---|------|---------------------------|------|
| 1 | strict + 全真 where，x=[1,2], extra=[3,4] | `COLUMN_NOT_IN_SCHEMA(extra)`；where/check 各 1 次 | 一致 |
| 2 | ordered + where，输入 y,x | `COLUMN_NOT_ORDERED`，坏值 y | 一致 |
| 3 | coerce，x=['1','bad','3'] | `DATATYPE_COERCION` 保留 bad，且有 `WRONG_DATATYPE` | 一致（另有下游比较 CHECK_ERROR） |
| 4 | 列 parser 删末行 + 独立 y 检查 | `CHECK_ERROR(x)` + `DATAFRAME_CHECK(y)` | 一致 |
| 5 | 列 parser 抛普通异常 | `CHECK_ERROR`，`original_exc=ValueError`，定位 x，y 仍跑 | 一致 |
| 6 | parser 失败 + drop_invalid_rows | 仍抛 `SchemaErrors` | 一致 |
| 7 | 仅普通行错误 + drop | 精确剔除，保留 [2] | 一致 |

其它已验证组合（脚本在 `~/work/repro/`）：全假 where（check_fn 0 次但结构错误仍报）、
重复普通索引/MultiIndex 的物理位置剔除、`n_failure_cases` 仅限量展示不影响剔除、
warning 不阻断、合法 parser 重排/筛行/改值、宽表 100 错误不持有整表、
regex 多列各自坏值（bad1/bad2 → v1/v2）、pickle、两次调用不串错误、
SeriesSchema、Index、scalar 非法输出。

## 4. 复跑命令

环境（本机无 pip，先在用户态引导并建 venv）：

```bash
python3 /tmp/get-pip.py --user --break-system-packages
python3 -m pip install --user --break-system-packages virtualenv
python3 -m virtualenv ~/envs/pandera
~/envs/pandera/bin/python -m pip install "numpy==1.26.4" "pandas==2.2.3" \
    pydantic typeguard typing_extensions typing_inspect packaging \
    hypothesis pyyaml pytest "pytest-asyncio<0.25" pytest-xdist joblib scipy \
    setuptools_scm
~/envs/pandera/bin/python -m pip install -e . --no-build-isolation
```

受影响回归（全部通过）：

```bash
~/envs/pandera/bin/python -m pytest \
    tests/pandas tests/base tests/dask/test_dask_not_installed.py \
    tests/test_inspection_utils.py \
    --ignore=tests/pandas/test_docs_setting_column_widths.py \
    --ignore=tests/pandas/test_pandas_engine.py \
    --ignore=tests/pandas/test_pyarrow_engine.py
# => 2082 passed, 3 skipped, 4 xfailed
```

新增专项测试（35 项）：

```bash
~/envs/pandera/bin/python -m pytest tests/pandas/test_error_retention.py -q
# => 35 passed
```

策略/属性测试：

```bash
~/envs/pandera/bin/python -m pytest tests/strategies -q
# => 513 passed, 4 xfailed
```

人工表脚本：

```bash
~/envs/pandera/bin/python -I ~/work/repro/expectations_table.py
```

## 5. 真实结果

- pandas + base 等可运行目录：**2082 passed, 3 skipped, 4 xfailed**。
- 新增 `tests/pandas/test_error_retention.py`：**35 passed**。
- 两个旧测试按新契约更新：
  - `test_row_changing_column_parser_is_rejected`：由期望
    `SchemaDefinitionError` 改为期望聚合的 `CHECK_ERROR`（且依赖检查不产生记录）。
  - `test_drop_invalid_for_dataframe_schema`：dtype 不匹配原本“删光所有行”，
    现拆出 `test_drop_invalid_rows_raises_on_dtype_mismatch`，断言抛
    `SchemaErrors`（结构错误不能靠删行修复）。

## 6. 未验证项（环境限制）

- `pyarrow` 与 NumPy 1.26.4 不兼容（pyarrow 新版要求 NumPy≥2），因此
  `test_pandas_engine.py` / `test_pyarrow_engine.py` 未运行。
- `sphinx` 缺失，`test_docs_setting_column_widths.py` 未运行（仅文档构建相关）。
- `dask` / `modin` / `polars` / `pyspark` / `geopandas` 重型后端未安装，对应集成
  测试未运行；但它们复用 pandas backend 的共享代码路径，且
  `test_dask_not_installed.py` 通过。polars/pyspark 有独立 backend，本次未改其文件。
- 线程安全由现有 `test_where_checks_are_thread_safe` 与 `test_multithreaded.py` 覆盖。
