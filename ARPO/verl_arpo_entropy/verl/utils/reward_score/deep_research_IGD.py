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
        solution_str: The solution string to evaluate
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
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    bonus = 0.0

    
    # 随机决定是否打印（1/64 的概率）
    do_print = random.randint(1, 64) == 1
    
    response = solution_str
    valid_template, reason = validate_format(response)
    
    #  【情况1】 格式错误全为-1
    if not valid_template:
        if do_print:
            print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score_format"] = -1
        result["score"] = result["score_format"] + result["score_accuracy"] + result["score_mi"]
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
        result["score"] = result["score_format"] + result["score_accuracy"] + result["score_mi"]
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
        result["score"] = result["score_format"] + result["score_accuracy"] + result["score_mi"]
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
        
        # if extra_info is not None and "mi_reward_increment" in extra_info:
        #     mi_reward_increment = float(extra_info["mi_reward_increment"])
        #     result["score_mi"] = mi_reward_increment
        #     if do_print:
        #         print(f"score_mi: {mi_reward_increment:.4f}")

    # 总分 = 格式分数 + 准确率分数 + 第三个分数 + bonus
    result["score"] = result["score_format"] + (result["score_accuracy"]+bonus) + result["score_mi"]*0.4
    if do_print:
        print(f"score_format: {result['score_format']:.4f}")
        print(f"score_accuracy+bonus: {result['score_accuracy']+bonus:.4f}")
        print(f"score_mi*0.4: {result['score_mi']*0.4:.4f}")

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

