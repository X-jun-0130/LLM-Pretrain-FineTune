# ==================== TTRL: GenRM 选择模式 ====================
SELECTION_PROMPT_TEMPLATE = '''
你是一名严谨的医学专家，正在对多个 AI 生成的候选回答进行同行评议，从多个候选答案中选出**最优答案**。

**任务：同行评议与最佳答案遴选**
    - **零容忍原则**：任何包含危险医疗建议、严重术语错误或逻辑断裂的回答必须直接淘汰。
    - **严禁根据答案长度进行判断**。短小精悍的回答可能优于冗长空洞的回答。

**评估维度（请综合考量）**
1.  **答案准确性**：评估答案是否合理、符合用户意图。
2.  **逻辑严密性**：推理链条是否完整？前后是否存在矛盾？是否回答了用户问题的核心诉求？
3.  **语言表达**：评估答案的语言表达是否流畅自然、是否存在不当的语言混杂、格式是否清晰可控。

<评估信息>
**对话历史**（上下文信息，按时间顺序排列，从最早到最新）
```
{插入对话历史}
```

**当前问题**（用户提出的具体请求）
```
{插入原始问题}
```

**知识核查清单**
```
### 核心知识（均来自权威书籍与指南）
{插入核心知识}
```

**候选答案列表**（共 {n_candidates} 个候选答案）
{candidates_block}
</评估信息>


**输出格式**
请进行对比分析，选出最优答案，当最优答案有多个时，请返回所有最优答案的序号。
**注意：必须仅输出对应的数字序号列表，严禁输出答案文本内容。**
将结果严格置于 `\\boxed{[选中的答案序号列表]}` 中，如 \\boxed{[1,3,5]}。
若所有答案质量均差，请返回 \\boxed{[]}。请勿返回其他内容。
'''.strip()


def get_selection_prompt(
    message: list[dict],
    candidates: list[str],
) -> str:
    """构建 GenRM 选择模式的 prompt
    
    Args:
        message: 原始对话消息列表（raw_prompt）
        candidates: 候选答案列表（n 个 response）
        core_knowledge: 核心知识（可选）
    Returns:
        构建好的选择模式 prompt
    """
    prompt = SELECTION_PROMPT_TEMPLATE
    
    # 对话历史和问题
    if len(message) >= 3:
        chathistory = message[:-1]
        history = '\n'.join([k['role'] + '：' + k['content'].strip() for k in chathistory])
        prompt = prompt.replace('{插入对话历史}', history)
        question = 'user：' + message[-1]['content'].strip()
    else:
        prompt = prompt.replace('{插入对话历史}', '无')
        question = '\n'.join([k['role'] + '：' + k['content'].strip() for k in message])
    
    prompt = prompt.replace('{插入原始问题}', question.strip())
    
    # 候选答案块
    prompt = prompt.replace('{n_candidates}', str(len(candidates)))
    candidates_parts = []
    for i, cand in enumerate(candidates, 1):
        # 截断过长的候选答案以控制 prompt 长度
        cand_text = cand.strip()
        candidates_parts.append(f"--- 候选答案 {i} ---\n```\n{cand_text}\n```")
    prompt = prompt.replace('{candidates_block}', '\n\n'.join(candidates_parts))
    
    return prompt



check_prompt :str = '''
你是一名专业的评估专家，需根据以下核心要素来对「预测答案」进行质量评估：

- **当前问题**（用户提出的具体请求）
- **预测答案**（待评估的答案）

### ⭐ 评分标准【评分应极其严格与严谨】：
**评估维度（请综合考量）--- 满分100分**
1.  **答案准确性**：评估答案是否合理、符合用户意图。
2.  **语言表达**：评估答案的语言表达是否流畅自然、是否存在不当的语言混杂、格式是否清晰可控。
4.  **其他维度**：答案是否包含分析过程再给出JSON格式答案；JSON格式是否带有markdown标记。
--- 

### ❓ 当前问题
```
{插入原始问题}
```

### 🤖 预测答案（待评估答案）
```
{插入待评估答案}
```

---

请按照以下结构输出你的评估结果：

### 📊 评估分析

[在此处进行逐项对比分析]

### 📌 预测答案评估分数

\\boxed{预测答案评估总分}
'''
