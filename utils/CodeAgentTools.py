import black
import subprocess
import re
import os
from typing import Tuple, Union
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import ast
import textwrap
from utils.utils import copy_file, recover

class CodeInterpreter:
    def __init__(self, repo_path: str, conda_env: str, gpu_id: int):
        """
        Initialize the Code Interpreter.

        :param repo_path: Path to the code repository.
        :param conda_env: The name of the Anaconda environment to use.
        :param gpu_id: The ID of the GPU to assign for running the code.
        """
        self.repo_path = repo_path
        self.conda_env = conda_env
        self.gpu_id = gpu_id
        PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(PY_LANGUAGE)

        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path '{repo_path}' does not exist.")
        print(f"Code Interpreter initialized with repo: {repo_path}, "
              f"Conda environment: {conda_env}, GPU ID: {gpu_id}")

    def extract_error_lines(self, stderr_str):
        """
        Return only lines containing typical Python exception messages
        or the traceback lines themselves.
        """
        # Some possible keywords to keep (you can customize):
        patterns = [
            r'^Traceback \(most recent call last\):',
            r'^\s*File\s+\"',
            r'^(\w+Error|Exception):',
            r'^.*Run:.*$',
            r'^.*Error:.*$',  # added to catch error lines that don't match the others
        ]

        keep_regex = re.compile('|'.join(patterns))
        # We'll collect lines that match any of the above patterns
        keep_lines = []

        for line in stderr_str.splitlines():
            if keep_regex.search(line):
                keep_lines.append(line)
            # Also keep lines following a traceback (heuristic: if the line is indented after a 'File ' line)
            elif keep_lines and line.startswith('    '):
                # It's likely part of the traceback stack
                keep_lines.append(line)

        # If we found anything, return it, otherwise return entire stderr
        return "\n".join(keep_lines) if keep_lines else stderr_str

    def run_code(self, CodeRepoPath: str, file_name: str, command: str) -> Tuple[bool, Union[str, Exception]]:
        file_path = os.path.join(self.repo_path, file_name)
        
        if not os.path.exists(file_path):
            return False, f"File '{file_name}' does not exist in the repository."

        process = None
        try:
            # Prepare the command - fix nested quotes issue
            cmd = (
                f"cd {CodeRepoPath} && "
                f"/bin/bash -c '"
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {self.conda_env} && "
                f"export CUDA_VISIBLE_DEVICES={self.gpu_id} && "
                f"{command} --TestCode'"
            )

            # Use Popen instead of run to get access to the PID
            process = subprocess.Popen(
                cmd,
                shell=True,
                text=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                cwd=self.repo_path
            )
            
            # Access the process ID
            pid = process.pid
            print(f"Process started with PID: {pid}")
            
            # Set a timeout
            try:
                stderr = process.communicate(timeout=60*15)[1]
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                print(f"Process {pid} timed out. Killing process and its children...")
                self._kill_process_tree(pid)
                process.wait()
                return False, "Execution timed out after 15 minutes. Far too long!"
            
            if returncode == 0:
                self._kill_process_tree(pid)
                return True, ""
            else:
                self._kill_process_tree(pid)
                return False, self.extract_error_lines(stderr)
        except Exception as e:
            if process is not None and process.poll() is None:
                # Process is still running, kill it
                pid = process.pid
                print(f"Exception occurred. Killing process {pid} and its children...")
                self._kill_process_tree(pid)
            return False, str(e)

    def _kill_process_tree(self, pid):
        """Kill a process and all its children to ensure GPU resources are released."""
        try:
            # First try to get all child processes
            child_pids_cmd = f"pgrep -P {pid}"
            try:
                child_pids = subprocess.check_output(child_pids_cmd, shell=True, text=True)
                for child_pid in child_pids.strip().split('\n'):
                    if child_pid:  # Check if not empty
                        self._kill_process_tree(int(child_pid))
            except subprocess.SubprocessError:
                # No children or pgrep failed
                pass
                
            # Kill the main process
            os.kill(pid, 9)  # SIGKILL
            print(f"Process {pid} killed successfully")
        except Exception as e:
            print(f"Error killing process {pid}: {str(e)}")
    
    def FormatCheckTool(self, code_string):
        """
        Formats the provided Python code string using Black and returns the formatted version.

        Args:
            code_string (str): The Python code to format.

        Returns:
            str: The formatted code or an error message.
        """
        try:
            # Format the code using Black
            formatted_code = black.format_str(code_string, mode=black.FileMode())
            return formatted_code
        except black.NothingChanged:
            return code_string  # Code is already properly formatted
        except Exception as e:
            return f"An error occurred during formatting: {e}"

    def extract_function_name_from_code(self, code):
        """
        Extracts the function name from a Python function definition using Tree-sitter.
        """
        tree = self.parser.parse(code.encode('utf-8'))
        root_node = tree.root_node

        for child in root_node.children:
            if child.type == "function_definition":
                function_name_node = child.child_by_field_name("name")
                if function_name_node:
                    return function_name_node.text.decode("utf-8")
        raise ValueError("No function definition found in the provided code.")

    def replace_function_in_file(self, file_path, new_definition):
        """
        Replaces a function definition in a Python file with a new one using Tree-sitter.
        """
        # Extract the function name from the new definition
        function_name = self.extract_function_name_from_code(new_definition)
        print(f"Replacing function: {function_name}")

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Parse the file content to locate the function
        tree = self.parser.parse(content.encode('utf-8'))
        root_node = tree.root_node
        updated_content = content
        for child in root_node.children:
            if child.type == "function_definition":
                function_name_node = child.child_by_field_name("name")
                if function_name_node and function_name_node.text.decode("utf-8") == function_name:
                    # Extract the start and end byte positions of the function
                    start_byte = child.start_byte
                    end_byte = child.end_byte

                    # Replace the old function with the new definition
                    updated_content = content[:start_byte] + new_definition + content[end_byte:]
                    # print(updated_content)
                    break

        # Write the updated content back to the file
        # updated_content = self.FormatCheckTool(updated_content)
        output_file_path = file_path.replace(".py", "_updated.py")
        with open(output_file_path, 'w') as file:
            file.write(updated_content)

class DependencyVisitor(ast.NodeVisitor):
    """
    Visits nodes in a function (or method) body and collects
    names of functions/methods that are called.
    """
    def __init__(self):
        self.dependencies = set()

    def visit_Call(self, node):
        # node.func can be a Name (e.g., func()) or an Attribute (e.g., obj.method())
        if isinstance(node.func, ast.Name):
            self.dependencies.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            full_name = get_full_attr_name(node.func)
            if full_name:
                self.dependencies.add(full_name)
        # Continue traversing inside the call node.
        self.generic_visit(node)

def find_external_apis(code):
    """
    Identify external APIs used in the code, resolving aliases to their original module paths.

    Args:
        code (str): Python code string to analyze.

    Returns:
        list: List of fully qualified external API names.
    """
    class ExternalAPIFinder(ast.NodeVisitor):
        def __init__(self):
            self.alias_map = {}  # Maps alias names to their original fully qualified names.
            self.external_apis = set()

        def visit_Import(self, node):
            for alias in node.names:
                original_name = alias.name
                alias_name = alias.asname or alias.name
                self.alias_map[alias_name] = original_name
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            # Skip wildcard imports.
            if node.names[0].name != '*':
                module_name = node.module
                for alias in node.names:
                    original_name = f"{module_name}.{alias.name}"
                    alias_name = alias.asname or alias.name
                    self.alias_map[alias_name] = original_name
            self.generic_visit(node)

        def visit_Call(self, node):
            full_name = self.get_full_name(node.func)
            if full_name:
                leftmost = full_name.split('.')[0]
                if leftmost in self.alias_map:
                    replaced = self.replace_alias_with_original(full_name)
                    self.external_apis.add(replaced)
            self.generic_visit(node)

        def get_full_name(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                prefix = self.get_full_name(node.value)
                if prefix:
                    return prefix + "." + node.attr
            return None

        def replace_alias_with_original(self, full_name):
            """
            Attempt to resolve full_name using the alias map. In particular, if the last component
            (e.g. "CrossEntropyLoss") is imported, that mapping takes precedence.
            """
            parts = full_name.split('.')
            # If the entire full_name exactly exists in the alias map, return its mapping.
            if full_name in self.alias_map:
                return self.alias_map[full_name]
            # If the last part was imported directly, return that mapping.
            if parts[-1] in self.alias_map:
                return self.alias_map[parts[-1]]
            # Otherwise, if the first part is an alias, replace it.
            if parts[0] in self.alias_map:
                original_base = self.alias_map[parts[0]]
                original_parts = original_base.split('.')
                # Only skip duplicate if the next part equals the last part of original_base.
                if len(parts) > 1 and parts[1] == original_parts[-1]:
                    if len(parts) > 2:
                        return original_base + '.' + '.'.join(parts[2:])
                    else:
                        return original_base
                else:
                    return original_base + '.' + '.'.join(parts[1:])
            return full_name

    tree = ast.parse(code)
    finder = ExternalAPIFinder()
    finder.visit(tree)
    return list(finder.external_apis)

def extract_import_statements(code_str):
    """
    从给定的 Python 代码字符串中提取所有 import 语句。

    参数:
        code_str (str): Python 代码字符串
    返回:
        list: 包含所有 import 语句的列表（字符串形式）
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        print("代码解析出错:", e)
        return []
    
    import_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
    
    import_statements = []
    for node in import_nodes:
        # 尝试还原源代码中的原始导入语句（要求 Python 3.8+）
        segment = ast.get_source_segment(code_str, node)
        if segment:
            import_statements.append(segment.strip())
        else:
            # 如果无法获取原始语句，则手动构造一条简单的表示
            if isinstance(node, ast.Import):
                names = ", ".join(alias.name for alias in node.names)
                import_statements.append(f"import {names}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                names = ", ".join(alias.name for alias in node.names)
                import_statements.append(f"from {module} import {names}")
    
    return import_statements

def find_local_function_calls(code, target_function_name):
    """
    找到目标函数中调用的、在同一文件中定义的其他函数。
    
    参数:
        code (str): Python代码字符串
        target_function_name (str): 目标函数的名称
    
    返回:
        list: 目标函数中调用的本地函数名称列表
    """
    # 第一步：收集所有顶层函数定义并找到目标函数节点
    class LocalFunctionFinder(ast.NodeVisitor):
        def __init__(self, target_name):
            self.target_name = target_name
            self.local_functions = set()  # 存储文件中定义的顶层函数
            self.target_node = None      # 存储目标函数的AST节点

        def visit_FunctionDef(self, node):
            # 记录所有顶层函数名称
            self.local_functions.add(node.name)
            # 如果是目标函数，保存其AST节点
            if node.name == self.target_name:
                self.target_node = node
            # 不遍历函数体，只收集顶层定义

    # 第二步：分析目标函数中的函数调用
    class CallFinder(ast.NodeVisitor):
        def __init__(self, local_functions):
            self.local_functions = local_functions
            self.called_functions = set()  # 存储目标函数中调用的本地函数

        def visit_Call(self, node):
            # 检查调用是否为本地函数
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.local_functions:
                    self.called_functions.add(func_name)
            self.generic_visit(node)

    # 解析代码为AST
    tree = ast.parse(code)
    
    # Phase 1: 收集所有顶层函数定义和目标函数节点
    finder = LocalFunctionFinder(target_function_name)
    finder.visit(tree)
    
    # 如果目标函数未找到，抛出异常
    if finder.target_node is None:
        raise ValueError(f"Function {target_function_name} not found in the code.")
    
    # Phase 2: 分析目标函数中的函数调用
    call_finder = CallFinder(finder.local_functions)
    for stmt in finder.target_node.body:
        call_finder.visit(stmt)
    
    return list(call_finder.called_functions)

def find_method_dependencies(class_code, method_name):
    """
    Find class variables and methods that a specified method depends on.

    Args:
        class_code (str): The class definition as a string.
        method_name (str): The name of the target method.

    Returns:
        tuple: (list of dependent variables, list of dependent methods)
    """
    class DependencyFinder(ast.NodeVisitor):
        def __init__(self, method_name):
            self.method_name = method_name
            self.class_variables = set()  # Class variables defined
            self.class_methods = set()    # Class methods defined
            self.dependencies_vars = set()  # Variables used by target method
            self.dependencies_methods = set()  # Methods called by target method
            self.current_method = None    # Current method being analyzed

        def visit_ClassDef(self, node):
            # Step 1: Collect all method names
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    self.class_methods.add(child.name)
                    if child.name == self.method_name:
                        self.target_method = child
            # Step 2: Collect class variables from all methods
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    self.current_method = child
                    for stmt in child.body:
                        self.visit(stmt)
            # Step 3: Analyze the target method for dependencies
            if hasattr(self, 'target_method'):
                self.current_method = self.target_method
                for stmt in self.target_method.body:
                    self.visit(stmt)
            self.current_method = None

        def visit_Assign(self, node):
            # Collect class variable definitions (self.xxx = ...)
            if self.current_method:
                for target in node.targets:
                    if (isinstance(target, ast.Attribute) and 
                        isinstance(target.value, ast.Name) and 
                        target.value.id == 'self'):
                        self.class_variables.add(target.attr)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # Collect class variables and methods used in the target method
            if (self.current_method and 
                self.current_method.name == self.method_name):
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == 'self'):
                    if node.attr in self.class_methods:
                        self.dependencies_methods.add(node.attr)
                    elif node.attr in self.class_variables:
                        self.dependencies_vars.add(node.attr)
            self.generic_visit(node)

        def visit_Call(self, node):
            # Collect method calls and check arguments for self.xxx in the target method
            if (self.current_method and 
                self.current_method.name == self.method_name):
                # Check if this is a method call (self.xxx())
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'self'):
                    if node.func.attr in self.class_methods:
                        self.dependencies_methods.add(node.func.attr)
                # Check arguments for self.xxx
                for arg in node.args:
                    if (isinstance(arg, ast.Attribute) and 
                        isinstance(arg.value, ast.Name) and 
                        arg.value.id == 'self'):
                        if arg.attr in self.class_variables:
                            self.dependencies_vars.add(arg.attr)
                        elif arg.attr in self.class_methods:
                            self.dependencies_methods.add(arg.attr)
            self.generic_visit(node)

    # Parse the class code and analyze it
    tree = ast.parse(class_code)
    finder = DependencyFinder(method_name)
    finder.visit(tree)
    return list(finder.dependencies_vars), list(finder.dependencies_methods)

def extract_class_definition(code, class_name):
    """
    从给定的Python代码字符串中抽取指定类的定义。
    
    参数:
        code (str): Python文件的内容（字符串）
        class_name (str): 要抽取的类名（字符串）
    
    返回:
        str: 指定类的定义代码，如果未找到则返回None
    """
    # 将代码解析为抽象语法树（AST）
    tree = ast.parse(code)
    
    # 遍历AST，查找匹配的类定义
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # 获取类的起始行号（注意：ast.lineno从1开始，列表索引从0开始）
            start_line = node.lineno - 1
            # 获取结束行号（Python 3.8+支持end_lineno，否则手动计算）
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else find_end_line(code, start_line)
            # 将代码按行分割并提取指定范围
            lines = code.splitlines()
            class_code = '\n'.join(lines[start_line:end_line])
            return class_code
    
    # 如果未找到类，返回None
    return None

def find_end_line(code, start_line):
    """
    辅助函数：找到类定义的结束行（适用于Python 3.7及以下）。
    
    参数:
        code (str): Python代码字符串
        start_line (int): 类定义的起始行号（基于0的索引）
    
    返回:
        int: 类定义的结束行号
    """
    lines = code.splitlines()
    indent = None
    for i in range(start_line, len(lines)):
        line = lines[i]
        # 确定类定义的缩进级别
        if indent is None and line.strip().startswith('class '):
            indent = len(line) - len(line.lstrip())
        # 当遇到与类同级或更少的缩进时，类定义结束
        elif indent is not None:
            if not line.strip() or len(line) - len(line.lstrip()) <= indent:
                return i
    return len(lines)

def get_node_source(node, lines):
    """
    Given an AST node and the file's source split into lines,
    return the source code snippet corresponding to the node.
    """
    if hasattr(node, "end_lineno"):
        return "\n".join(lines[node.lineno - 1: node.end_lineno])
    else:
        return lines[node.lineno - 1]

def get_full_attr_name(node):
    """
    Recursively obtain a full attribute name from an AST Attribute node.
    For example, for a call like: module.submodule.func(),
    this function returns "module.submodule.func".
    """
    if isinstance(node, ast.Attribute):
        value = get_full_attr_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        else:
            return node.attr
    elif isinstance(node, ast.Name):
        return node.id
    else:
        return None

def get_dependencies(func_node):
    """
    Given a function or method AST node, return a set of dependency names
    (i.e. names of called functions or methods) found in its body.
    """
    visitor = DependencyVisitor()
    visitor.visit(func_node)
    return visitor.dependencies

def analyze_file(filename, targetfile=False):
    with open(filename, "r", encoding="utf-8") as file:
        source = file.read()
    
    lines = source.splitlines()
    functions = []   # List of dicts: { "name": ..., "definition": ..., "dependencies": [...] }
    classes = []     # List of dicts: { "name": ..., "definition": ..., "methods": [...] }
    global_vars = [] # Mapping of variable name -> definition snippet
    imports = []
    try:
        tree = ast.parse(source, filename)
             # List of import statements (as strings
        for node in tree.body:
            # Top-level function definitions.
            if isinstance(node, ast.FunctionDef):
                snippet = get_node_source(node, lines)
                deps = get_dependencies(node)
                functions.append({
                    "name": node.name,
                    "definition": snippet,
                    "path": str(filename),
                    "dependencies": sorted(list(deps)),
                    "visited": targetfile
                })
            
            # Top-level class definitions.
            elif isinstance(node, ast.ClassDef):
                snippet = get_node_source(node, lines)
                class_methods = []
                # Process methods within the class.
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_snippet = get_node_source(child, lines)
                        deps = get_dependencies(child)
                        class_methods.append({
                            "name": child.name,
                            "definition": method_snippet,
                            "path": str(filename),
                            "dependencies": sorted(list(deps)),
                            "visited": targetfile
                        })
                classes.append({
                    "name": node.name,
                    "definition": snippet,
                    "path": str(filename),
                    "methods": class_methods,
                    "visited": targetfile
                })
            
            # Import statements.
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imp_str = f"import {alias.name}"
                    if alias.asname:
                        imp_str += f" as {alias.asname}"
                    imports.append(imp_str)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imp_str = f"from {module} import {alias.name}"
                    if alias.asname:
                        imp_str += f" as {alias.asname}"
                    imports.append(imp_str)
            
            # Global variable assignments.
            elif isinstance(node, ast.Assign):
                snippet = get_node_source(node, lines)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in global_vars:
                            tmp = dict()
                            tmp['name'] = target.id
                            tmp["path"] = str(filename)
                            tmp['definition'] = snippet
                            tmp['visited'] = targetfile
                            global_vars.append(tmp)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name) and elt.id not in global_vars:
                                tmp = dict()
                                tmp['name'] = elt.id
                                tmp["path"] = str(filename)
                                tmp['definition'] = snippet
                                tmp['visited'] = targetfile
                                global_vars.append(tmp)
            
            elif isinstance(node, ast.AnnAssign):
                snippet = get_node_source(node, lines)
                if isinstance(node.target, ast.Name):
                    if node.target.id not in global_vars:
                        tmp = dict()
                        tmp['name'] = node.target.id
                        tmp["path"] = str(filename)
                        tmp['definition'] = snippet
                        tmp['visited'] = targetfile
                        global_vars.append(tmp)
            
            elif isinstance(node, ast.AugAssign):
                snippet = get_node_source(node, lines)
                if isinstance(node.target, ast.Name):
                    if node.target.id not in global_vars:
                        tmp = dict()
                        tmp['name'] = node.target.id
                        tmp["path"] = str(filename)
                        tmp['definition'] = snippet
                        tmp['visited'] = targetfile
                        global_vars.append(tmp)
    except Exception as e:
        print(f"Error parsing file {str(filename)}: {e}")

    return functions, classes, global_vars, imports
 
def remove_space(code_list):
    for m in range(len(code_list)-1, -1, -1):
        if code_list[m] == '':
            code_list.pop(m)
        else:
            code_list[m] = code_list[m].replace('\n', '')
    return code_list

def run_code(code, data, task_id, CodeRepoPath, CondaEnvName, gpu_id, file_path, Command, Benchmark_path_global, conda_env_path, reference=False):
    task_data = data['task_details'][task_id]
    CondaEnv = os.path.join(conda_env_path, CondaEnvName)
    InterpreterTool = CodeInterpreter(CodeRepoPath, CondaEnv, gpu_id)
    if not reference:
        code = textwrap.indent(code, '    ' * (task_data['indent']-1))
    Code_File_Ori = task_data['ori_python_file'].split('\n')
    start_line = task_data['signature_position'][0]
    end_line = task_data['body_position'][1]
    del Code_File_Ori[start_line-1:end_line]
    code_gen_list = code.split('\n\n')
    Code_File_New = Code_File_Ori
    Code_File_New[start_line:start_line] = code_gen_list
    file_new_string = '\n'.join(Code_File_New)
    recover(data, task_id, Benchmark_path_global)
    with open(file_path, 'w') as f:
        f.write(file_new_string)
    Success, feedback=InterpreterTool.run_code(CodeRepoPath, file_path, Command.strip())
    return Success, feedback


