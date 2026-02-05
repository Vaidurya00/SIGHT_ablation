import re
import sys
import string
import os
# Fix: Ensure standard library math is imported before local math.py
# Python adds script's directory to sys.path[0], causing local math.py to shadow stdlib math
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
# Remove local math from sys.modules if already imported
if 'math' in sys.modules:
    math_file = getattr(sys.modules['math'], '__file__', None)
    if math_file and _script_dir in math_file:
        del sys.modules['math']
# Import standard library math first
import math
# Now import random (which needs math from stdlib)
import random
# Restore script directory to sys.path for other local imports
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer

# 从环境变量读取格式奖励专用步数阈值
FORMAT_ONLY_STEPS = int(os.environ.get("FORMAT_ONLY_STEPS", "0"))


def validate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"

        current_pos = result_end_pos


    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"





def is_valid_sequence(text):
    '''
    验证文本是否符合要求的标签序列。下面按「可能的输出路径」说明允许的写法。

    ---------- 可能的输出路径 ----------

    路径 1（不搜、直接答）：
        <think> 推理内容 </think> <answer> \\boxed{答案} </answer>
    即：一轮 think 后直接给 answer，中间不能插 search/result。

    路径 2（搜一轮再答）：
        <think> 推理 </think> <search> 查询 </search> <result> 检索结果 </result>
        <self-evidence> 提炼的证据 </self-evidence>
        <think> 继续推理 </think> <answer> \\boxed{答案} </answer>
    即：第一轮 think → search → result 后，必须先有一对 <self-evidence></self-evidence>，
    再才能写下一轮 <think>，然后才能 <answer>。
    注：self-evidence 必须有一对标签，但标签中间的内容可以为空（可写 <self-evidence></self-evidence>）。

    路径 3（多轮搜索）：
        <think> ... </think> <search>...</search> <result>...</result>
        [<hint> 可选 hint </hint>]
        <self-evidence>...</self-evidence>  <think> ... </think> <search>...</search> <result>...</result>
        <self-evidence>...</self-evidence>  <think> ... </think> <answer> \\boxed{答案} </answer>
    规则：除第一轮外，每一轮 <think> 前面都必须先有一对 <self-evidence></self-evidence>；
    search 和 result 必须成对出现；hint 只能出现在 result 之后、self-evidence 之前，且可选。

    路径 4（有 hint 时）：
        ... </result> <hint> 提示 </hint> <self-evidence>...</self-evidence> <think> ...
    即：result 之后可以接 hint，再接 self-evidence，再接下一轮 think。

    通用约束：
    - <answer> 必须恰好出现一次，且内容里包含 \\boxed{...}。
    - 所有标签（think/search/result/answer/hint/self-evidence）开闭数量一致。
    - 标签之间除空白外不能夹其它内容（正文只能写在标签内部）。
    - 各标签内部的内容（think/search/result/hint/self-evidence/answer）在格式校验上均可为空；
      只有 <answer> 会额外要求内容里出现 \\boxed{...}，故 answer 不能整段为空。
    '''
    content = text
    
    # Check for balanced tags (including self-evidence: 除去第一轮，之后每次 think 前要有 <self-evidence></self-evidence>)
    tags_to_check = ["think", "search", "result", "answer", "hint", "self-evidence"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Check <answer> tag count: must have exactly one pair
    answer_opening_count = len(re.findall(r"<answer>", content))
    answer_closing_count = len(re.findall(r"</answer>", content))
    if answer_opening_count != 1 or answer_closing_count != 1:
        return False, f"<answer> tag must appear exactly once: found {answer_opening_count} opening and {answer_closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    # 除去第一轮：只有 start 可直接到 <think>；之后每轮必须 after_self_evidence -> <think>
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|result|answer|hint|self-evidence)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> ... -> answer -> end; 多轮时 after_result/hint -> self_evidence -> think -> ...
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|result|answer|hint|self-evidence)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "after_self_evidence"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<result>" and state == "after_search":
                state = "in_result"
            elif part == "</result>" and state == "in_result":
                state = "after_result"
            elif part == "<hint>" and state == "after_result":
                state = "in_hint"
            elif part == "</hint>" and state == "in_hint":
                state = "after_hint"
            elif part == "<self-evidence>" and state in ["after_result", "after_hint"]:
                state = "in_self_evidence"
            elif part == "</self-evidence>" and state == "in_self_evidence":
                state = "after_self_evidence"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_result", "in_answer", "in_hint", "in_self_evidence"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "after_result", "after_hint", "after_self_evidence"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
    
    # Check if \boxed{} is in the answer
    answer_start = content.find('<answer>')
    answer_end = content.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = content[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"
        
    return True, "Valid sequence format"




def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer content from the text within <answer> tags.
    
    Args:
        text: The text to extract answer from
        
    Returns:
        Optional[str]: The extracted answer or None if no match
    """
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)


def extract_self_evidence_combined(text: str) -> Optional[str]:
    """
    Extract all content from <self-evidence> </self-evidence> tags and concatenate.
    Reference: qa_em.py extract_information - 拼在一起.
    
    Returns:
        Combined string or None if no matches.
    """
    pattern = r"<self-evidence>(.*?)</self-evidence>"
    matches = re.findall(pattern, text.strip(), re.DOTALL)
    if not matches:
        return None
    combined = " ".join(m.strip() for m in matches).strip()
    return combined if combined else None


def self_evidence_reward(text: str, ground_truth: Union[str, List[str]]) -> float:
    """
    Self-evidence reward: 拼接所有 <self-evidence>...</self-evidence> 内容，
    若包含 ground truth 答案给 0.2，否则 0。参考 qa_em.py cover_em_check(information, target).
    """
    combined = extract_self_evidence_combined(text)
    if combined is None:
        return 0.0
    norm_combined = normalize_answer(combined)
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    for gt in ground_truth:
        if normalize_answer(gt) in norm_combined:
            return 0.2
    return 0.0



def extract_solution_test(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    print(matches)
    print(len(matches))
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    ans =  matches[-1].group(1).strip()
    boxed_content = last_boxed_only_string(ans)
    if boxed_content is not None:
        ans = remove_boxed(boxed_content)
    # If there are 2 or more matches, return the last one
    return ans




def remove_boxed(s: str) -> str:
    """
    Remove the LaTeX \boxed{} wrapper from the string.
    
    Args:
        s: String potentially containing \boxed{}
        
    Returns:
        str: String with \boxed{} removed
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from the string.
    
    Args:
        string: String to extract \boxed{} from
        
    Returns:
        Optional[str]: The extracted \boxed{} content or None if not found
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s: str) -> str:
    """
    Normalize the answer string by removing articles, white spaces, punctuation and converting to lowercase.
    
    Args:
        s: String to normalize
        
    Returns:
        str: Normalized string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    Calculate F1 score between prediction and ground truths.
    
    Args:
        prediction: The predicted answer
        ground_truths: The ground truth answer(s)
        
    Returns:
        float: F1 score
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if  (precision + recall)  > 0 else 0.0

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.
    
    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate  (只有回答没有prompt)
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information
        
    Returns:
        Dict[str, Any]: A dictionary containing the score and additional information
    """

    # 初始化统一的返回结构
    result = {
        "score": 0,
        "score_format": 0,      # 格式分数
        "score_accuracy": 0,     # 准确率分数（F1）
        "score_mi": 0,          # 第三个分数（预留接口）
        "score_self_evidence": 0,  # self-evidence 奖励：有答案给 0.2
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    bonus = 0.0
    # breakpoint()
    # 随机决定是否打印（1/64 的概率）
    do_print = random.randint(1, 64) == 1
    
    response = solution_str
    valid_template, reason = is_valid_sequence(response)
    
    #  【情况1】 格式错误全为-1
    if not valid_template:
        if do_print:
            print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score_format"] = -1
        result["score"] = result["score_format"] + result["score_accuracy"] 
        result["reason"] = f"bad format: {reason}"
        return result
    
    
    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    # 【情况2】基本格式正确但答案格式错误
    answer_part = extract_answer(response)
    if answer_part is None:
        if do_print:
            print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score_format"] = -0.5
        result["score"] = result["score_format"] + result["score_accuracy"] 
        result["reason"] = "cannot extract answer"
        return result
    
    # 【情况2】基本格式正确但答案格式错误
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        if do_print:
            print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score_format"] = -0.5
        result["score"] = result["score_format"] + result["score_accuracy"]
        result["reason"] = f"find box error: {e}"
        return result
    
    # 获取当前训练步数（从extra_info中获取，如果没有则默认为0）
    current_step = 0
    if extra_info is not None and "global_step" in extra_info:
        current_step = int(extra_info["global_step"])
    
    # 检查是否在前N步（只使用format reward）
    use_format_only = FORMAT_ONLY_STEPS > 0 and current_step < FORMAT_ONLY_STEPS
    
    if use_format_only:
        # 前N步：只计算format reward，其他reward设为0
        result["reason"] = f"format_only_mode (step {current_step} < {FORMAT_ONLY_STEPS})"
        if do_print:
            print(f"[Format-Only Mode] Step {current_step}: Only format reward is active, accuracy and MI rewards are set to 0")
    else:
        # 正常模式：计算所有reward
        # 【情况3】 格式正确的前提下就看f1（0-1） 然后调用了工具+0.1
        f1_score = get_f1_score(answer, ground_truth)
        result["f1_score"] = f1_score
        result["score_accuracy"] = f1_score
        if do_print:
            print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")
        
        
        # 计算总分（三个 score 之和）
        if f1_score > 0 and "</search>" in response:
            if do_print:
                print(f"--------------------------------correct answer with tool call--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
            bonus = 0.1
            result["reason"] = f"correct answer and calling search, get score: {f1_score + bonus}"
        elif f1_score > 0:
            if do_print:
                print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
            result["reason"] = f"correct answer, get f1 score: {f1_score}"
        else:
            if do_print:
                print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
            result["reason"] = f"wrong answer but good format: {answer}"
        

    # 总分 = 格式分数 + 准确率分数 + 第三个分数 + bonus
    result["score"] = result["score_format"] + (result["score_accuracy"]+bonus) + result["score_self_evidence"]

    if do_print:
        print(f"score_format: {result['score_format']:.4f}")
        print(f"score_accuracy+bonus: {result['score_accuracy']+bonus:.4f}")
        print(f"score_self_evidence: {result['score_self_evidence']:.4f}")
        print(f"score: {result['score']:.4f}")


    return result


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct")
    extra_info = {"tokenizer": tokenizer}

    response = "<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.<|im_end|>\n<|im_start|>user\nIn each cell, a strip of length $100$ is worth a chip. You can change any $2$ neighboring chips and pay $1$ rouble, and you can also swap any $2$ chips for free, between which there are exactly $4$ chips. What is the smallest amount of rubles you can spend to rearrange the chips in reverse order?<|im_end|>\n<|im_start|>assistant\n<think> To solve the given problem, we will use both natural language reasoning and Python code. Let's break down the problem step-by-step.\n\n1. **Given Equations:**\n   \\[\n   \\log_{10}(\\sin x) + \\log_{10}(\\cos x) = -1\n   \\]\n   Using the properties of logarithms, we can combine the logs:\n   \\[\n   \\log_{10}(\\sin x \\cos x) = -1 \\implies \\sin x \\cos x = 10^{-1} = \\frac{1}{10}\n   \\]\n   Recall the double-angle identity for sine: \\(\\sin(2x) = 2 \\sin x \\cos x\\). So,\n   \\[\n   \\sin(2x) = 2 \\left(\\frac{1}{10}\\right) = \\frac{1}{5}\n   \\]\n\n2. **Second Given Equation:**\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}(\\log_{10}(n) - 1)\n   \\]\n   We can rewrite the right-hand side using properties of logarithms:\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}\\log_{10}(n) - \\frac{1}{2} = \\log_{10}(n^{1\/2}) - \\log_{10}(10^{1\/2}) = \\log_{10}\\left(\\frac{\\sqrt{n}}{\\sqrt{10}}\\right)\n   \\]\n   This implies:\n   \\[\n   \\sin x + \\cos x = \\frac{\\sqrt{n}}{\\sqrt{10}}\n   \\]\n   Squaring both sides, we get:\n   \\[\n   (\\sin x + \\cos x)^2 = \\frac{n}{10}\n   \\]\n   Expanding the left side:\n   \\[\n   \\sin^2 x + \\cos^2 x + 2 \\sin x \\cos x = \\frac{n}{10}\n   \\]\n   Using the Pythagorean identity \\(\\sin^2 x + \\cos^2 x = 1\\) and the earlier result \\(\\sin x \\cos x = \\frac{1}{10}\\):\n   \\[\n   1 + 2 \\left(\\frac{1}{10}\\right) = \\frac{n}{10} \\implies 1 + \\frac{1}{5} = \\frac{n}{10} \\implies \\frac{6}{5} = \\frac{n}{10} \\implies n = 12\n   \\]\n\nNow, let's verify this with Python code to ensure the result is accurate.\n </think><python>\nimport math\r\nfrom sympy import symbols, Eq, solve, log, sin, cos\r\n\r\n# Define the variable\r\nx = symbols('x')\r\n\r\n# First equation: log10(sin(x)) + log10(cos(x)) = -1\r\neq1 = Eq(log(sin(x), 10) + log(cos(x), 10), -1)\r\n\r\n# Solve for sin(x) * cos(x)\r\nsin_x_cos_x = solve(eq1, sin(x) * cos(x))[0]\r\nsin_x_cos_x = sin_x_cos_x.simplify()\r\n\r\n# We know sin(x) * cos(x) = 1\/10\r\nsin_x_cos_x_value = 1 \/ 10\r\n\r\n# Second equation: log10(sin(x) + cos(x)) = 1\/2 * (log10(n) - 1)\r\nn = symbols('n')\r\neq2 = Eq(log(sin(x) + cos(x), 10), (1\/2) * (log(n, 10) - 1))\r\n\r\n# We know (sin(x) + cos(x))^2 = 1 + 2 * sin(x) * cos(x) = 1 + 2 * (1\/10) = 6\/5\r\nsin_x_plus_cos_x_squared = 6 \/ 5\r\nsin_x_plus_cos_x = math.sqrt(sin_x_plus_cos_x_squared)\r\n\r\n# Substitute sin(x) + cos(x) into the second equation\r\nlhs = log(sin_x_plus_cos_x, 10)\r\nrhs = (1\/2) * (log(n, 10) - 1)\r\n\r\n# Solve for n\r\nn_value = solve(Eq(lhs, rhs), n)[0]\r\nprint(n_value)\n</python><result>\n\n12.0000000000000\n</result><search> query </search><result> result </result><answer>The value of \\( n \\) is \\(\\boxed{12}\\). The calculations and the verification using Python both confirm that the final answer is indeed \\( n = 12 \\).</answer><|im_end|>"
    answer = "12"
    res = compute_score("test_data_source", response, answer, extra_info)
    print(res)

    response2 = "To find the theatre in London Borough of Hackney that had a performance of The Birth of Merlin in the 1600s, let\'s begin by identifying venues historically associated with the 1600s in the London Borough of Hackney. \n\n<search> theatres in London Borough of Hackney 1600s </search> <result>\nDoc 1 (Title: \"Hackney Theatre\")\nThe Hackney Theatre, located in the London Borough of Hackney, has a rich history of performances dating back to the 1600s. One notable production was \'The Birth of Merlin,\' which was performed during this period. The theatre has been a significant cultural hub in Hackney, showcasing a variety of theatrical works and contributing to the local arts scene.\n\nDoc 2 (Title: \"London Borough of Hackney\")\nby connecting it to other parts of the city and stimulating development. London\'s first Tudor theatres were built at Shoreditch. The Gunpowder Plot was first exposed nearby in Hoxton. In 1727 Daniel Defoe said of the villages of Hackney The parish church of St John-at-Hackney was built in 1789, replacing the nearby former 16th-century parish church dedicated to St Augustine (pulled down in 1798). Notable residents from the 17th, 18th and 19th centuries included Robert Aske, William Cecil, Samuel Courtauld, Samuel Hoare, Joseph Priestley and Thomas Sutton. Many grand houses stood in Stoke Newington and Stamford Hill; the latter neighbourhood\n\nDoc 3 (Title: \"Curtain Theatre\")\nCurtain Theatre The Curtain Theatre was an Elizabethan playhouse located in Hewett Street, Shoreditch (part of the modern London Borough of Hackney), just outside the City of London. It opened in 1577, and continued staging plays until 1624. The Curtain was built some south of London\'s first playhouse, The Theatre, which had opened a year before, in 1576. It was called the ""Curtain"" because it was located near a plot of land called Curtain Close, not because of the sort of front curtain associated with modern theatres, but of its proximity of the City walls, curtain or curtain wall referring\n</result><|im_end|>"
    answer2 = "3"    
    res2 = compute_score("test_data_source", response2, answer2, extra_info)
    print(res2)

    response3 = "Let's approach this step-by-step. First, we need to determine which super villain the alleged sister of Zaladane appeared as. The key is to find out the alias or role of this character that was associated with Zaladane's sister.\\n\\n<search query}>Zaladane sister super villain</search>\\n\\n <result>\\nDoc 1 (Title: \\\"Sister of Zaladane\\\")\\nThe sister of Zaladane is known as the character known as 'Sister'. In the comics, she has been portrayed as a supervillain who shares a complex relationship with Zaladane. Their interactions often revolve around themes of power, betrayal, and family dynamics within the mutant community.\\n\\nDoc 2 (Title: \\\"Super Villain Identity\\\")\\nThe sister of Zaladane has appeared as the supervillain known as 'Malice'. Malice is a character that embodies the darker aspects of mutant powers and has been a formidable foe to various superhero teams, including the X-Men.\\n\\nDoc 3 (Title: \\\"Zaladane\\\")\\nZaladane is a fictional character appearing in American comic books published by Marvel Comics. She is a supervillain and a member of the mutant community. Zaladane is known for her ability to manipulate and control the powers of other mutants, particularly those related to the earth and its elements. She has been involved in various storylines, often opposing the X-Men and other superhero teams.\\n</result> \\n\\n<think>Based on the search results, the sister of Zaladane appeared as the supervillain Malice.</think>\\n\\n<answer>\\\\boxed{Malice}</answer>"
    answer3 = "Malice"
    res3 = compute_score("test_data_source", response3, answer3, extra_info)
    print(res3)

    response4 = "system\\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\\\[ \\\\boxed{answer here} \\\\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\\\boxed{} with latex format.\\nuser\\nWhat cellist, and musician, associated with the experimental music venue The Kitchen is one of the artists in the music of Iowa?\\nassistant\\n<think> I need to find a cellist who is associated with The Kitchen and is also associated with the music of Iowa. </think> <search> cellist associated with The Kitchen and music of Iowa </search> <result>\\nDoc 1 (Title: \\\"Lindsay Mac\\\")\\nLindsay Mac Lindsay Mac is a singer/songwriter/cellist raised in Iowa City, Iowa and based in Cambridge, Massachusetts. Mac began playing the cello at the age of nine. She attended Dartmouth College, and her interest in becoming a classical cellist led her to study abroad at the Royal College of Music in London. She briefly left Dartmouth after being accepted to the San Francisco Conservatory of Music. While there, she had concluded that classical music was not her calling. Returning to receive her bachelor's degree from Dartmouth, and living in a wood-heated New Hampshire cabin, Mac began adapting her cello technique.\\n\\nDoc 2 (Title: \\\"Sven Hansell\\\")\\nSven Hansell Sven Hostrup Hansell (23 October 1934 – 6 March 2014) was an American musicologist and Professor Emeritus of Musicology at the University of Iowa. He was a specialist in the music and performance practices of the 17th and 18th centuries, as well as a harpsichordist and composer. Hansell was born in New York City and grew up in Philadelphia. He received his bachelor's degree from the University of Pennsylvania in 1956 and a Master's Degree from Harvard University in 1958. He then studied composition with Nadia Boulanger in France as well as undertaking further studies at the University\\n\\nDoc 3 (Title: \\\"Steven Doane\\\")\\nSteven Doane Steven Doane (born December 10, 1950) is an American cellist and recitalist. Internationally acclaimed as a soloist, chamber musician, and master teacher, Steven Doane maintains an active performance schedule throughout the U.S and Europe. Starting cello at a young age, his principal teachers included Richard Kapuscinski, Bernard Greenhouse, Jane Cowan (in Edrom), and Janos Starker. A finalist in the 1974 International Tchaikovsky Competition, Doane originally entered on a dare from friends. From this point in his career, he went on to serve as principal cellist of the Milwaukee Symphony Orchestra (1976–77), and later the Rochester Philharmonic Orchestra (1981–83).\\n</result>\\n<hint>You have previously searched for 'cellist associated with The Kitchen and music of Iowa'. This did not yield the answer. Analyze the gap between the current tool result and the final goal. What is missing? Based on the analysis, generate a new search query that targets the missing information.</hint>\\nBased on the information provided, there isn't a direct match for a cellist associated with The Kitchen and music of Iowa. However, The Kitchen is an experimental music venue in New York City. Let's search for a cellist associated with The Kitchen who is also linked to the music of Iowa.\\n<search> cellist associated with The Kitchen and music of Iowa </search> <result>\\n\\n</result>\\n<hint>You have previously searched for 'cellist associated with The Kitchen and music of Iowa'. This did not yield the answer. You must generate a semantically distinct query now.</hint>\\nBased on the search results, there isn't a clear match for a cellist associated with The Kitchen and music of Iowa. However, one can infer that the cellist and musician associated with The Kitchen and music of Iowa could be Sven Hansell. Let's verify this.\\n<think> I need to verify if Sven Hansell is a cellist and is associated with The Kitchen and music of Iowa. </think> <search> Sven Hansell cellist and associated with The Kitchen and music of Iowa </search> <result>\\n\\n</result>\\n<hint>You have previously searched for 'Sven Hansell cellist and associated with The Kitchen and music of Iowa'. This did not yield the answer. You must generate a semantically distinct query now.</hint>\\nAfter searching, there is no clear evidence that Sven Hansell is a cellist or is associated with The Kitchen and music of Iowa. Given the constraints of the information available, it seems there might be a mix-up in the search parameters. However, based on the best available information, the cellist and musician associated with the experimental music venue The Kitchen who is also associated with the music of Iowa could be inferred as Sven Hansell, although this inference is not strongly supported by the available data.\\n<answer>\\\\boxed{Sven Hansell}</answer>"
    answer4 = "Malice"
    res4 = compute_score("test_data_source", response4, answer4, extra_info)
    print(res4)

    res5 = extract_solution_test(response4)
    print(res5) 
