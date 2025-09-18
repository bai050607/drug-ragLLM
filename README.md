### 项目概览

本项目基于 Neo4j 图谱与文本检索（RAG）能力，实现对病历文本进行相关信息检索，返回推荐药物。核心能力：
- 从 Neo4j 图数据库中拉取节点构建向量索引，结合 LLM 进行检索与生成。
- 支持一次性读取并缓存药物集合，对模型输出做集合内过滤与去重，保证结果只来自候选集合。
- 批处理 JSONL 病历文件，输出指定格式的预测结果文件。

### 目录结构（关键文件）
- `raggraph.py`：图检索与药物推荐核心类 `DrugGraph`。
- `work.py`：批处理入口，读取 JSONL 输入，调用流程并生成提交结果。
- `qianwen_class.py`：封装千问 LLM 与 Embedding 的适配类。
- `候选药物列表.json`：候选药物名称列表（中文字符串数组）。
- `submit_pred.json`：运行后生成的结果文件（包含 ID 与 prediction）。

### 配置
- Neo4j 连接：默认使用 `bolt://localhost:7687` / 用户 `neo4j` / 密码 `12345678`。
  - 如需更改，请直接修改 `work.py` 中 `DrugGraph` 初始化参数，或在 `raggraph.py` 中自定义。
- 候选药物文件：
  - 代码默认读取当前目录的 `候选药物列表.json`。
  - 也可通过环境变量 `CANDIDATE_DRUGS_JSON` 指定绝对路径。
- 千问 API：
  - 设置环境变量 `DASHSCOPE_API_KEY`。
  - 其它参数见 `qianwen_class.py`。

### 输入与输出
- 输入：JSONL（每行一个 JSON 对象）。示例字段：
  - `就诊标识`：作为样本 ID；
  - `诊疗过程描述`（或 `medical_text`/`text`/`query`/`input`）：作为病历文本。
- 输出：`submit_pred.json`，数组格式，每项：
```json
{
  "ID": "1-1",
  "prediction": ["阿司匹林肠溶片", "氯吡格雷"]
}
```

### 运行说明
1) 准备好 Neo4j 服务和候选药物文件：
   - 确保 Neo4j 可用，并且存在可拉取的节点（至少包含 `name` 属性）。
   - 确保有 `候选药物列表.json`（中文名称字符串数组）。

2) 设置环境变量（可选）：
```bash
export DASHSCOPE_API_KEY=你的千问Key
export CANDIDATE_DRUGS_JSON=/root/drug-ragLLM/候选药物列表.json
```

3) 执行批处理：
```bash
python /root/drug-ragLLM/work.py
```

4) 结果文件：
   - 运行结束后在当前目录生成 `submit_pred.json`。

### 注意事项
- 项目默认以中文名称为准，请保证候选集合与图谱中的药物名称口径一致；
- 如需扩大检索范围或调整索引规模，修改 `raggraph.py` 中的 Neo4j 拉取逻辑（`LIMIT` 与文本拼接）。

### 免责声明
本项目仅用于技术实验与演示，非临床诊疗系统。任何推荐结果仅供参考，不构成医疗建议。


