import json
import os, re
import textwrap
from codebleu import calc_codebleu
from utils.utils import remove_space, insert_code_before_returns, run_compare, Extract_Function_Definition, remove_comments_from_function
from utils.CodeAgentTools import find_external_apis, extract_import_statements, find_local_function_calls, find_method_dependencies, extract_class_definition, run_code
from utils.Reason_Process_ACC import calculate_reason_process_acc_multi
import argparse
import pickle   
import shutil
from sklearn.metrics import recall_score
from pathlib import Path

def FormatCode(code):
    code = code.strip()
    for m in range(len(code)):
        if code[m:m+9] == "```python":
            code = code[m+9:]
            break
    for m in range(len(code)-1, 0, -1):
        if code[m-2:m+1] == "```":
            code = code[:m-2]
            break
    return code

def remove_comments(function_str):
    """
    Remove triple-quoted strings and single-line comments from the given function string.
    """
    if not isinstance(function_str, str):
        raise TypeError("Input to remove_comments must be a string")
    # Remove triple-quoted strings (docstrings and multi-line strings)
    function_str = re.sub(r'"""[\s\S]*?"""', '', function_str)
    # Remove single-line comments starting with #
    function_str = re.sub(r'#.*', '', function_str)
    return function_str

def process_code_lines(code_lines):
    """
    Process a list of code lines, removing invalid lines and comments.
    """
    valid_lines = []
    for line in code_lines:
        try:
            # Check if the line is valid and can be processed
            remove_comments(line)
            valid_lines.append(line)
        except TypeError:
            print(f"Invalid line removed: {line.strip()}")
    # Join the remaining valid lines into a single string
    valid_code = ''.join(valid_lines)
    # Remove comments and docstrings from the remaining code
    cleaned_code = remove_comments(valid_code)
    return cleaned_code

def extract_scores(file_path):
    """
    Extracts scores from a text file containing lines with the pattern:
    "Benchmark Dir: <path>, Task: <number>, Score: <score>"
    
    :param file_path: Path to the text file
    :return: List of extracted scores as floats
    """
    scores = []
    pattern = re.compile(r"Score:\s*([0-9]*\.?[0-9]+)")
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                scores.append(float(match.group(1)))

    mean_value = sum(scores) / len(scores)
    return mean_value

def count_true_in_file(file_path):
    """
    统计文件中有多少个 "True"
    
    参数:
        file_path (str): 文件路径
    
    返回:
        int: True 的数量
    """
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 统计 "True" 的出现次数（区分大小写）
            count = content.count('True')
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return 0
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return 0
    
    return count/100

def codeBLEU_Score(args):
    root_path = args.root_path
    result_path = args.result_path
    Data_path = os.path.join(root_path, "Data.json")
    Benchmark_path = os.path.join(root_path, "Benchmark")
    Benchmark_list = [d for d in os.listdir(Benchmark_path) 
                     if os.path.isdir(os.path.join(Benchmark_path, d))]
    Dict_Direct_25 = dict()
    Dict_Direct_25['codebleu'] = 0
    Dict_Direct_25['ngram_match_score'] = 0
    Dict_Direct_25['weighted_ngram_match_score'] = 0
    Dict_Direct_25['syntax_match_score'] = 0
    Dict_Direct_25['dataflow_match_score'] = 0
    Num_task = 0
    
    with open(Data_path, 'r') as f:
        Data = json.load(f)
    for k in range(0, len(Benchmark_list)):
        benchmark_dir = ""
        for tmp in Benchmark_list:
            if tmp.startswith(str(k)+ '-'):
                benchmark_dir = tmp
                benchmark_dir = os.path.join(root_path, benchmark_dir)
                break
        if benchmark_dir == "":
            break

        if args.model == 'o3-mini' or args.model == 'o1':
            OutputDir = os.path.join(result_path, str(k), "SciReproducer_" +  args.model+ "_" + args.effort) 
        else:  
            OutputDir = os.path.join(result_path, str(k), "SciReproducer_" + args.model) 
        
        data = Data[k]
        for i in range(0, len(data['task_details'])):
            output_file = os.path.join(OutputDir,f"task{str(i+1)}" + ".pkl")
            Result = pickle.load(open(output_file, 'rb'))
            Answer = Result['answer']
            Num_task += 1
            tmp = dict()
            task = data['task_details'][i]
            ReferencePythonFile = Data[k]['task_details'][i]['ori_python_file']
            ReferenceCodeWithComments = task['ReferenceCode_With_Comments']
            ReferenceCodeWithComments  = textwrap.indent(ReferenceCodeWithComments, '    ' * task['indent'])
            ReferenceCode = Extract_Function_Definition(task, ReferencePythonFile)
            ReferenceCode += ReferenceCodeWithComments
            ReferenceCode = remove_comments_from_function(ReferenceCode)
            Answer = remove_comments_from_function(Answer)
            resultNoPaper_25 = calc_codebleu([ReferenceCode], [Answer], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
            for key in resultNoPaper_25.keys():
                Dict_Direct_25[key] += resultNoPaper_25[key]
    
    for key in resultNoPaper_25.keys():
        Dict_Direct_25[key] /= Num_task
    print(Dict_Direct_25)

def execution_ACC(args):
    root_path = args.root_path
    result_path = args.result_path
    Data_path = os.path.join(root_path, "Data.json")
    
    if args.model == 'o3-mini' or args.model == 'o1':
        output_exetution_file_path = os.path.join(result_path, "ExecutionACC", "SciReproducer_" + args.model + "_" + args.effort + "_result.txt")
    else:
        output_exetution_file_path = os.path.join(result_path, "ExecutionACC", "SciReproducer_" + args.model + "_result.txt")
    
    gpu_id = args.gpu_id
    with open(Data_path, 'r') as f:
        Data = json.load(f)

    for k in range(0, 36):
        data = Data[k]
        repo_path = data['project_path']
        CodeRepoPath = os.path.join(root_path, repo_path)
        CondaEnvName = data['enviorment_name']
        
        if args.model == 'o3-mini' or args.model == 'o1':
            output_dir = os.path.join(result_path, str(k), "Run", "SciReproducer_" + args.model+ "_" + args.effort) 
            gen_code_path = os.path.join(result_path, str(k), "SciReproducer_" + args.model+ "_" + args.effort) 
        else:  
            output_dir = os.path.join(result_path, str(k), "Run", "SciReproducer_" + args.model) 
            gen_code_path = os.path.join(result_path, str(k), "SciReproducer_" + args.model)

        if args.reference:
            output_dir = os.path.join(result_path, str(k), "Run", "ReferenceCode")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        
        for i in range(0, len(data['task_details'])):
            task = data['task_details'][i]
            Python_File = Data[k]['task_details'][i]['ori_python_file']
            ReferenceCode = Extract_Function_Definition(task, Python_File)
            ReferenceCodeWithComments = task['ReferenceCode_With_Comments']
            ReferenceCodeWithComments  = textwrap.indent(ReferenceCodeWithComments, '    ' * task['indent'])
            ReferenceCode += ReferenceCodeWithComments
            if not args.reference:
                output_file = os.path.join(gen_code_path, f"task{str(i+1)}" + ".pkl")
                GenCodeDict = pickle.load(open(output_file, 'rb'))
                GenCodeDictAnswer = GenCodeDict['answer']
                code = GenCodeDictAnswer
            else:
                code = ReferenceCode

            code_gen_list = code.split('\n')
            code_gen_list = insert_code_before_returns(code_gen_list, i, output_dir)
            for line in range(len(code_gen_list)):
                if code_gen_list[line].endswith("\n"):
                    code_gen_list[line] = code_gen_list[line][:-1]
            code_after_insert = '\n'.join(code_gen_list)

            ReferencePythonFile = Data[k]['task_details'][i]['ori_python_file']
            ReferencePythonFile_line = ReferencePythonFile.split('\n')
            ReferencePythonFile_line = remove_space(ReferencePythonFile_line)
            start_line = task['signature_position'][0]
            end_line = task['body_position'][1]
            del ReferencePythonFile_line[start_line:end_line]
            

            Code_File_New = ReferencePythonFile_line
            for line in range(len(Code_File_New)):
                if Code_File_New[line].endswith("\n"):
                    Code_File_New[line] = Code_File_New[line][:-1]
            Code_File_New[start_line:start_line] = code_gen_list
            Code_File_New = '\n'.join(Code_File_New)

            Python_File_path = root_path + repo_path + task['completion_path'][1:]
            Success, feedback = run_code(code_after_insert, data, i, CodeRepoPath, CondaEnvName, gpu_id, Python_File_path, task['script'], root_path, args.conda_env_path, reference=args.reference)
            if not Success:
                print(feedback)
            else:  
                print(f"Repo: {k} Task: {i} Success")

        if not args.reference:
            script = f"python {CodeRepoPath}/compare.py --root_path {args.root_path} --result_root_path {args.result_path} --gen_code_path {gen_code_path} --OutputPath {output_exetution_file_path} "
            Success, feedback = run_compare(CodeRepoPath, "compare.py", script, CodeRepoPath, CondaEnvName, gpu_id, args.conda_env_path)
            if not Success:
                print(feedback)
            else:
                print(1)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    
    score = count_true_in_file(output_exetution_file_path)
    print(f"Execution ACC: {score}")

def Recall(args):
    root_path = args.root_path
    result_path = args.result_path
    Data_path = os.path.join(root_path, "Data.json")
    Benchmark_path = os.path.join(root_path, "Benchmark")
    Benchmark_list = [d for d in os.listdir(Benchmark_path) 
                     if os.path.isdir(os.path.join(Benchmark_path, d))]
    with open(Data_path, 'r') as f:
        Data = json.load(f)

    Num_task = 0

    api_all = list()
    local_all = list()
    cross_all = list()
    for k in range(0, 36):
        benchmark_dir = ""
        for tmp in Benchmark_list:
            if tmp.startswith(str(k)+ '-'):
                benchmark_dir = tmp
                benchmark_dir = os.path.join(root_path, benchmark_dir)
                break
        if benchmark_dir == "":
            break

        data = Data[k]
        if args.model == 'o3-mini' or args.model == 'o1':
            OutputDir = os.path.join(result_path, str(k), "SciReproducer_" +  args.model+ "_" + args.effort) 
        else:  
            OutputDir = os.path.join(result_path, str(k), "SciReproducer_" + args.model) 

        for i in range(0, len(data['task_details'])):
            task = data['task_details'][i]
            ReferenceCodeWithComments = task['ReferenceCode_With_Comments']
            ReferenceCodeWithComments  = textwrap.indent(ReferenceCodeWithComments, '    ' * 1)
            output_file = os.path.join(OutputDir,f"task{str(i+1)}" + ".pkl")
            ResultNoPaperDict = pickle.load(open(output_file, 'rb'))
            ResultNoPaperAnswer = ResultNoPaperDict['answer']
            
            Num_task += 1
            tmp = dict()
            task = data['task_details'][i]
            task_type = task['type']
            ReferencePythonFile = Data[k]['task_details'][i]['ori_python_file']
            ReferenceCode = Extract_Function_Definition(task, ReferencePythonFile)
            ReferenceCode += ReferenceCodeWithComments
            ResultNoPaperAnswer = FormatCode(ResultNoPaperAnswer)
            ResultNoPaperAnswer_tmp = ResultNoPaperAnswer
            import_str = '\n'.join(extract_import_statements(ReferencePythonFile))
            import_str += '\n'
            ReferenceCode = import_str + ReferenceCode
            ResultNoPaperAnswer = import_str + ResultNoPaperAnswer
            api_annotate = task['external_APIs']['list']
            intra_annotate = task['dependency']['intra_file']
            cross_annotate = task['dependency']['cross_file']
            try:
                gencode_api = find_external_apis(ResultNoPaperAnswer)
            except:
                gencode_api = list()

            ReferencePythonFile_lines = ReferencePythonFile.split('\n')
            function_definition = '\n'.join(ReferencePythonFile_lines[task['signature_position'][0]-1:task['signature_position'][1]])
            match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', function_definition)
            if match:
                function_name = match.group(1)  # 获取捕获组 1，即函数名
            
            start_line = task['signature_position'][0]
            end_line = task['body_position'][1]
            del ReferencePythonFile_lines[start_line-1:end_line]
            ResultNoPaperAnswer_tmp  = textwrap.indent(ResultNoPaperAnswer_tmp, '    ' * (task['indent']-1))
            ReferencePythonFile_lines[start_line:start_line] = ResultNoPaperAnswer_tmp.split('\n')
            PythonFile_new = '\n'.join(ReferencePythonFile_lines)
            try:
                local_func = find_local_function_calls(PythonFile_new, function_name)
            except:
                local_func = list()
            find = list()
    
            if task_type == 'method':
                class_name = task['namespace'].split('.')[-2]
                method_name = task['namespace'].split('.')[-1]
                try:
                    class_definition = extract_class_definition(PythonFile_new, class_name)
                    vars, methods = find_method_dependencies(class_definition, method_name)
                except:
                    vars = list()
                    methods = list()

                for h in range(len(vars)):
                    vars[h] = class_name + '.' + vars[h]
                for h in range(len(methods)):
                    methods[h] = class_name + '.' + methods[h]
                find = find + vars + methods

            find += local_func
            find += gencode_api

            for h in api_annotate:
                if h in find:
                    api_all.append(1)
                else:
                    api_all.append(0)

            for h in intra_annotate:
                if h in find:
                    local_all.append(1)
                else:
                    local_all.append(0)
            for h in cross_annotate:
                if h in find:
                    cross_all.append(1)
                else:
                    cross_all.append(0)

    # Fix recall calculation - parameters were in wrong order
    # api_all/local_all/cross_all contain the actual detection results (0 or 1)
    # We need to calculate recall properly: sum(detected) / total_dependencies
    
    recall_api = sum(api_all) / len(api_all) if len(api_all) > 0 else 0
    recall_local = sum(local_all) / len(local_all) if len(local_all) > 0 else 0  
    recall_cross = sum(cross_all) / len(cross_all) if len(cross_all) > 0 else 0

    print(f"Intra Recall: {recall_local}, Cross Recall: {recall_cross}, API Recall: {recall_api}")

def ReasoningGraph_ACC(args):
    root_path = args.root_path
    result_path = args.result_path
    Data_path = os.path.join(root_path, "Data.json")
    Benchmark_path = os.path.join(root_path, "Benchmark")
    Benchmark_list = [d for d in os.listdir(Benchmark_path) 
                     if os.path.isdir(os.path.join(Benchmark_path, d))]
    with open(Data_path, 'r') as f:
        Data = json.load(f)

    Num_task = 0

    if args.model == 'o3-mini':
        path = os.path.join(result_path, "ReasonACC" , "SciReproducer_" + args.model + "_" +  args.effort + ".txt")
    else:
        path = os.path.join(result_path, "ReasonACC" , "SciReproducer_" + args.model + ".txt")
        
    num = 0
    for k in range(0 , 36):
        benchmark_dir = ""
        for tmp in Benchmark_list:
            if tmp.startswith(str(k)+ '-'):
                benchmark_dir = tmp
                benchmark_dir = os.path.join(root_path, benchmark_dir)
                break
        if benchmark_dir == "":
            break
        data = Data[k]

        if args.model == 'o3-mini' or args.model == 'o1':
            OutputDir = os.path.join(result_path, str(k), "SciReproducer_" +  args.model+ "_" + args.effort) 
        else:  
            OutputDir = os.path.join(result_path, str(k), "SciReproducer_" + args.model)

        for i in range(0, len(data['task_details'])):
            task = data['task_details'][i]
            ReferenceCodeWithComments = task['ReferenceCode_With_Comments']
            ReferenceCodeWithComments  = textwrap.indent(ReferenceCodeWithComments, '    ' * task['indent'])
            output_file = os.path.join(OutputDir,f"task{str(i+1)}" + ".pkl")
            ResultDict = pickle.load(open(output_file, 'rb'))
            ResultAnswer = ResultDict['answer']

            Num_task += 1
            tmp = dict()
            task = data['task_details'][i]
            latex_code = task['latex_code']
            ReferencePythonFile = Data[k]['task_details'][i]['ori_python_file']

            ReferenceCode = Extract_Function_Definition(task, ReferencePythonFile)
            ReferenceCode += ReferenceCodeWithComments
            ResultAnswer = FormatCode(ResultAnswer)
            
            # Retry mechanism for score calculation with better error handling
            max_retries = 5
            success = False
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    score = calculate_reason_process_acc_multi(latex_code, ReferenceCode, ResultAnswer, num)
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    print(f"Attempt {attempt + 1}/{max_retries} failed for Benchmark {k}, Task {i}: {e}")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        import time
                        time.sleep(0.5)  # Brief pause before retry
            
            if success:
                string = f"Benchmark Dir: {benchmark_dir}, Task: {i}, Score: {score}\n"
                print(string.strip())
                path1 = Path(path)
                path1.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'a') as f:
                    f.write(string)
            else:
                error_string = f"Benchmark Dir: {benchmark_dir}, Task: {i}, Failed to calculate score after {max_retries} attempts, Final Error: {last_error}\n"
                print(error_string.strip())
                with open(path, 'a') as f:
                    f.write(error_string)
            
            num += 1  # Always increment counter regardless of success/failure
    
    score = extract_scores(path)
    print(f"Reasoning Graph ACC: {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='gpt-4o', choices=['gpt-4o', 'gpt-4o-mini', 'o3-mini', 'deepseek-r1', 'deepseek-v3', 'claude-3-7', 'gemini-2.0-flash', 'gemini-2.0-flash-thinking'])
    parser.add_argument('--effort', default="low", type=str)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--conda_env_path', default="/Users/yanzhengxiang/work/my_work/CodeGenSLU/envs_sci", type=str)
    parser.add_argument('--metric', default='CodeBLEU_Score', choices=['CodeBLEU_Score', 'execution_ACC', 'Recall', 'ReasoningGraph_ACC'], type=str)
    parser.add_argument('--root_path', default='/Users/yanzhengxiang/work/my_work/CodeGenSLU/', type=str)
    parser.add_argument('--result_path', default='/Users/yanzhengxiang/work/my_work/CodeGenSLU/Result', type=str)
    args = parser.parse_args()
    args.reference = True
    print(args.model)
    print(args.effort)
    if args.metric == 'CodeBLEU_Score':
        codeBLEU_Score(args)
        print("CodeBLEU Score Finished")
    elif args.metric == 'execution_ACC':
        execution_ACC(args)
        print("Execution ACC Finished")
    elif args.metric == 'Recall':
        Recall(args)
        print("Recall Finished")
    elif args.metric == 'ReasoningGraph_ACC':
        ReasoningGraph_ACC(args)
        print("Reasoning Graph ACC Finished")

