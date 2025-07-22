import ast
import re
from typing import Dict, List, Set, Optional
import textwrap
from utils.utils import FormatCheckTool
from collections import deque
from utils.utils import llm
import os

def correct_code(code):
    prompt = f"""
I will provide you with a code snippet extracted from a function. This snippet may contain syntax errors that prevent it from being parsed by Python's ast module. Your task is to fix only the syntax errors while preserving the rest of the code. You may add print(1) if necessary to resolve structural issues without altering functionality.

Code Snippet:
{code}

Please return the python code directly without any additional information. do not include string like '```python ```'.

Your Answer: (Return only the modified code snippet, without any extra information)
""" 
    output = llm(prompt, stop=None, model='o3-mini')
    return output

class CodeComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.num_lines = 0
        self.defined_variables = set()
        self.used_variables = set()
        self.function_calls = 0
        self.operations = 0

    def visit_Assign(self, node):
        # Count defined variables - handle multiple assignment patterns
        for target in node.targets:
            self._extract_assigned_names(target)
        self.generic_visit(node)
    
    def _extract_assigned_names(self, node):
        """Helper method to extract variable names from assignment targets"""
        if isinstance(node, ast.Name):
            self.defined_variables.add(node.id)
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            # Handle tuple/list unpacking: a, b = func()
            for elt in node.elts:
                self._extract_assigned_names(elt)
        elif isinstance(node, ast.Subscript):
            # Handle subscript assignment: arr[0] = value
            if isinstance(node.value, ast.Name):
                self.used_variables.add(node.value.id)
        elif isinstance(node, ast.Attribute):
            # Handle attribute assignment: obj.attr = value
            if isinstance(node.value, ast.Name):
                self.used_variables.add(node.value.id)

    def visit_Name(self, node):
        # Count used variables
        if isinstance(node.ctx, ast.Load):
            self.used_variables.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Count function calls
        self.function_calls += 1
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # Count binary operations
        self.operations += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        # Count augmented assignments as operations and extract variables
        self.operations += 1
        # The target is both used and assigned
        self._extract_assigned_names(node.target)
        if isinstance(node.target, ast.Name):
            self.used_variables.add(node.target.id)
        self.generic_visit(node)
    
    def visit_If(self, node):
        # Conditional logic adds complexity
        self.operations += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        # Loop adds complexity
        self.operations += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        # Loop adds complexity
        self.operations += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        # Exception handling adds complexity
        self.operations += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        # Function definition adds complexity
        self.operations += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        # Class definition adds complexity
        self.operations += 1
        self.generic_visit(node)

    def visit_Expr(self, node):
        # Check expressions for function calls
        self.generic_visit(node)

    def analyze(self, code):
        self.num_lines = len(code.strip().split('\n'))
        code_ori = FormatCheckTool(code)
        code = code_ori
        while True:
            try:
                tree = ast.parse(code)
                break
            except Exception as e:
                print(f"Failed to parse code: {e}")
                code = self.correct_code(code_ori)
                code = code.strip()
                for m in range(len(code)):
                    if code[m:m+9] == "```python":
                        code = code[m+9:]
                        break
                for m in range(len(code)-1, 0, -1):
                    if code[m-2:m+1] == "```":
                        code = code[:m-2]
                        break

        self.visit(tree)
        
        # Calculate additional complexity metrics
        total_variables = len(self.defined_variables.union(self.used_variables))
        complexity_score = (
            self.operations * 0.3 +  # Operations weight
            self.function_calls * 0.2 +  # Function calls weight  
            total_variables * 0.1 +  # Variables weight
            self.num_lines * 0.05  # Lines of code weight
        )
        
        return {
            "Number of lines": self.num_lines,
            "Number of defined variables": len(self.defined_variables),
            "Number of used variables": len(self.used_variables),
            "Total unique variables": total_variables,
            "Number of function calls": self.function_calls,
            "Number of operations": self.operations,
            "Complexity score": round(complexity_score, 2),
        }
    
    def correct_code(self, code):
        prompt = f"""
I will provide you with a code snippet extracted from a function. This snippet may contain syntax errors that prevent it from being parsed by Python's ast module. Your task is to fix only the syntax errors while preserving the rest of the code. You may add print(1) if necessary to resolve structural issues without altering functionality.

Code Snippet:
{code}

Please return the python code directly without any additional information. do not include string like '```python ```'.

Your Answer: (Return only the modified code snippet, without any extra information)
    """ 
        output = llm(prompt, stop=None, model='o3-mini')
        return output

class SnippetDependencyAnalyzer:
    def __init__(self, code_str: str):
        """
        A class to parse a single Python function containing multiple code snippets,
        build snippet boundaries, extract direct dependencies, and store the snippet
        comments (headers) that appear before each snippet.
        """
        self.code_str = code_str
        self.code_str = FormatCheckTool(self.code_str)
        self.lines = self.code_str.splitlines()

        # { "Snippet X": (start_line, end_line), ... }
        self.snippet_boundaries: Dict[str, (int, int)] = {}
        # line_to_snippet[line_num] = "Snippet X" or None
        self.line_to_snippet: Dict[int, Optional[str]] = {}

        # snippet_vars["Snippet X"] = {"assigned": set(), "used": set()}
        self.snippet_vars: Dict[str, Dict[str, Set[str]]] = {}

        # snippet_deps["Snippet X"] = set of snippet_ids on which snippet_id depends
        self.snippet_deps: Dict[str, Set[str]] = {}

        # var_last_assign[var_name] = set of snippet_ids that last assigned that variable
        self.var_last_assign: Dict[str, Set[str]] = {}

        # NEW: snippet_comments["Snippet X"] = full comment block describing snippet
        self.snippet_comments: Dict[str, str] = {}

        # NEW: snippet_comment_code["Snippet X"] = code string of commented lines *within* the snippet itself.
        self.snippet_comment_code: Dict[str, str] = {}
        self.snippet_comment_code_complexity = dict()

    def extract_snippet_boundaries(self):
        """
        Identify snippet boundaries by scanning for lines of the form:
          # [Begin Snippet X]
        and
          # [End Snippet X]
        We'll store them in self.snippet_boundaries and also fill line_to_snippet.

        In addition, we extract the preceding comment lines for each snippet
        and store them in self.snippet_comments[snippet_id].
        """
        begin_pat = re.compile(r"#\s*\[Begin\s+Snippet\s+(\d+)\]", re.IGNORECASE)
        end_pat   = re.compile(r"#\s*\[End\s+Snippet\s+(\d+)\]", re.IGNORECASE)
        delimiter = re.compile(r"#\s*-+\s*", re.IGNORECASE)

        current_snip = None
        current_start = None
        snippet_id_list = list()
        for i, line in enumerate(self.lines, start=1):
            begin_match = begin_pat.search(line)
            if begin_match:
                snippet_id = f"Snippet {begin_match.group(1)}"
                current_snip = snippet_id
                snippet_id_list.append(snippet_id)
                current_start = i
                comment_block = self._collect_comment_lines(i - 1)
                self.snippet_comments[snippet_id] = comment_block

                continue

            end_match = end_pat.search(line)
            if end_match and current_snip is not None:
                snippet_id = f"Snippet {end_match.group(1)}"
                if snippet_id == current_snip:
                    # Mark boundaries 
                    self.snippet_boundaries[current_snip] = (current_start, i)
                    # Mark line_to_snippet for these lines
                    for ln in range(current_start, i + 1):
                        self.line_to_snippet[ln] = current_snip
                current_snip = None
                current_start = None

        for i in range(len(snippet_id_list)):
            snippet_key = snippet_id_list[i]
            if snippet_key in self.snippet_boundaries:
                continue
            start = None
            FindEnd = False
            for j, line in enumerate(self.lines, start=1):
                end_match = end_pat.search(line)
                if end_match:
                    snippet_id = f"Snippet {end_match.group(1)}"
                    if snippet_id == snippet_id_list[i]:
                        FindEnd = True
                        break
            if not FindEnd:
                for j, line in enumerate(self.lines, start=1):
                    begin_match = begin_pat.search(line)
                    if begin_match:
                        snippet_id = f"Snippet {begin_match.group(1)}"
                        if snippet_id == snippet_id_list[i]:
                            start = j
                            continue
                        if start != None:
                            for k in range(start, len(self.lines)):
                                line = self.lines[k]
                                if delimiter.search(line):
                                    self.snippet_boundaries[snippet_key] = (start, k)
                                    for ln in range(start, k):
                                        self.line_to_snippet[ln] = snippet_key
                                    break
                            break  
                            
        # After we have snippet boundaries, collect the commented code inside each snippet
        self._collect_snippet_comment_code()

    def _collect_comment_lines(self, start_line: int) -> str:
        """
        Walk backwards from 'start_line' while lines begin with '#'
        or are blank (optional). Stop when we hit a non-comment/ non-blank line
        or the beginning of the file. Then return them in normal top-down order.
        """
        collected = []
        idx = start_line

        while idx > 0:
            text = self.lines[idx - 1].strip()
            if text.startswith('#') or text == '':
                collected.append(self.lines[idx - 1])
                idx -= 1
            else:
                break

        # Now collected is in reverse order, so flip it
        collected.reverse()
        # Join them into one string
        return "\n".join(collected)

    def _collect_snippet_comment_code(self):
        """
        For each snippet, gather all lines *inside* the snippet boundary that begin with '#'.
        Store in self.snippet_comment_code[snippet_id] as a single string.
        """
        for snippet_id, (start_line, end_line) in self.snippet_boundaries.items():
            snippet_comment_lines = []
            # Lines in [start_line, end_line], inclusive
            for ln in range(start_line, end_line + 1):
                text = self.lines[ln - 1]
                if not text.strip().startswith('#'):
                    snippet_comment_lines.append(text)
            
            code = "\n".join(snippet_comment_lines)
            analyzer = CodeComplexityAnalyzer()    
            complexity = analyzer.analyze(code)
            if complexity == None:
                continue
            self.snippet_comment_code_complexity[snippet_id] = complexity
            self.snippet_comment_code[snippet_id] = "\n".join(snippet_comment_lines)
            total = 0
            for key, value in complexity.items():
                total += value
            self.snippet_comment_code_complexity[snippet_id]['total'] = total

    def _init_structs(self):
        for snip_id in self.snippet_boundaries.keys():
            self.snippet_vars[snip_id] = {"assigned": set(), "used": set()}
            self.snippet_deps[snip_id] = set()

    def analyze_ast(self):
        """
        Parse the entire code into an AST and walk it with a custom visitor to track
        direct dependencies (snippet -> snippet) and variable usage.
        """
        try:
            tree = ast.parse(self.code_str)
        except SyntaxError:
            self.code_str = correct_code(self.code_str)
            print("Corrected code: \n", self.code_str)
            tree = ast.parse(self.code_str)

        visitor = _SnippetVisitor(
            line_to_snippet = self.line_to_snippet,
            snippet_vars = self.snippet_vars,
            snippet_deps = self.snippet_deps,
            var_last_assign = self.var_last_assign
        )
        visitor.visit(tree)

    def build_dependency_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for snip_id, deps in self.snippet_deps.items():
            deps.discard(snip_id)  # remove self
            graph[snip_id] = sorted(deps)
        return graph

    def run_analysis(self) -> Dict[str, List[str]]:
        self.extract_snippet_boundaries()
        self._init_structs()
        self.analyze_ast()
        return self.build_dependency_graph()

class _SnippetVisitor(ast.NodeVisitor):
    """
    On assignment in snippet S => var_last_assign[var] = {S}.
    On usage in snippet S => snippet_deps[S] |= var_last_assign[var].
    Also track snippet_vars[S]["assigned"] / ["used"].
    """

    def __init__(
        self,
        line_to_snippet: Dict[int, Optional[str]],
        snippet_vars: Dict[str, Dict[str, Set[str]]],
        snippet_deps: Dict[str, Set[str]],
        var_last_assign: Dict[str, Set[str]]
    ):
        super().__init__()
        self.line_to_snippet = line_to_snippet
        self.snippet_vars = snippet_vars
        self.snippet_deps = snippet_deps
        self.var_last_assign = var_last_assign

    def _get_snippet_id(self, node: ast.AST) -> Optional[str]:
        return self.line_to_snippet.get(getattr(node, 'lineno', None))

    def visit_Assign(self, node: ast.Assign):
        # 1) Visit RHS
        self.generic_visit(node)
        # 2) Get union of snippet sets from variables used in RHS
        rhs_set = self._collect_rhs_sets(node.value)
        snippet_id = self._get_snippet_id(node)

        for tgt in node.targets:
            self._handle_assignment_target(tgt, snippet_id, rhs_set)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        lhs_set = self._collect_rhs_sets(node.target)
        rhs_set = self._collect_rhs_sets(node.value)
        combined = lhs_set.union(rhs_set)
        snippet_id = self._get_snippet_id(node)
        self._handle_assignment_target(node.target, snippet_id, combined)

    def _handle_assignment_target(self, target: ast.AST, snippet_id: Optional[str], rhs_set: Set[str]):
        if isinstance(target, ast.Name):
            self._assign_var(target.id, snippet_id, rhs_set)
        elif isinstance(target, ast.Tuple):
            # handle tuple destructuring
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self._assign_var(elt.id, snippet_id, rhs_set)
        elif isinstance(target, ast.Subscript):
            base_var = get_base_name(target.value)
            if base_var:
                self._assign_var(base_var, snippet_id, rhs_set)

    def _assign_var(self, var_name: str, snippet_id: Optional[str], rhs_set: Set[str]):
        if snippet_id is not None:
            self.var_last_assign[var_name] = {snippet_id}
            self.snippet_vars[snippet_id]["assigned"].add(var_name)
        else:
            # assigned outside snippet => inherit from RHS
            self.var_last_assign[var_name] = set(rhs_set)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            snippet_id = self._get_snippet_id(node)
            self._mark_use(node.id, snippet_id)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Load):
            base_var = get_base_name(node.value)
            snippet_id = self._get_snippet_id(node)
            if base_var:
                self._mark_use(base_var, snippet_id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        snippet_id = self._get_snippet_id(node)
        base_var = get_base_name(node.value)
        if base_var:
            self._mark_use(base_var, snippet_id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        # detect mutating calls, e.g. x.append(...)
        if isinstance(node.func, ast.Attribute):
            base_var = get_base_name(node.func.value)
            snippet_id = self._get_snippet_id(node)
            mutators = {"append","extend","pop","remove","insert","update","clear"}
            if base_var and node.func.attr in mutators:
                self._assign_var(base_var, snippet_id, set())

    def _mark_use(self, var_name: str, snippet_id: Optional[str]):
        if snippet_id is None:
            return
        self.snippet_vars[snippet_id]["used"].add(var_name)
        last_set = self.var_last_assign.get(var_name, set())
        self.snippet_deps[snippet_id].update(last_set)

    def _collect_rhs_sets(self, node: ast.AST) -> Set[str]:
        collector = _RHSCollector(self.var_last_assign)
        collector.visit(node)
        return collector.union_set

class _RHSCollector(ast.NodeVisitor):
    """
    Gathers union of var_last_assign sets for any variables used in an expression (RHS).
    """
    def __init__(self, var_last_assign: Dict[str, Set[str]]):
        super().__init__()
        self.var_last_assign = var_last_assign
        self.union_set: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.union_set |= self.var_last_assign.get(node.id, set())
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Load):
            base_var = get_base_name(node.value)
            if base_var:
                self.union_set |= self.var_last_assign.get(base_var, set())
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        base_var = get_base_name(node.value)
        if base_var:
            self.union_set |= self.var_last_assign.get(base_var, set())
        self.generic_visit(node)

def get_base_name(node: ast.AST) -> Optional[str]:
    """
    For e.g. senOri[str(...)] => 'senOri', or senOri.keys() => 'senOri'.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return get_base_name(node.value)
    elif isinstance(node, ast.Subscript):
        return get_base_name(node.value)
    return None

def extract_comments_to_single_line(code):
    """
    Extracts comments from a Python code string and transfers them into a single line.
    
    The function looks for multi-line comment blocks that describe snippets,
    typically starting after a line of dashes and containing descriptive text.

    Args:
        code (str): Python code as a string.

    Returns:
        str: Single line containing all comments.
    """
    code = textwrap.dedent(code)
    lines = code.splitlines()

    extracted_comments = []
    in_comment_block = False
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if this is a dash separator line (start of comment block)
        if re.match(r'^\s*#\s*-+\s*$', original_line):
            in_comment_block = True
            continue
            
        # If we're in a comment block and this is a comment line
        if in_comment_block and line.startswith("#"):
            # Remove leading '#' and whitespace
            comment_text = line.lstrip('#').strip()
            
            # Skip empty comment lines and lines that are just markers
            if (comment_text and 
                not re.match(r'^\[Begin\s+Snippet\s+\d+\]$', comment_text) and
                not re.match(r'^\[End\s+Snippet\s+\d+\]$', comment_text)):
                
                # If there's a colon, take the part after it
                if ':' in comment_text:
                    comment_text = comment_text[comment_text.find(":")+1:].strip()
                
                if comment_text:  # Only add non-empty comments
                    extracted_comments.append(comment_text)
                    
        # If we hit a non-comment line, we're no longer in the comment block
        elif in_comment_block and not line.startswith("#"):
            in_comment_block = False

    # Combine all comments into a single line, separated by a space
    return " ".join(extracted_comments)

def count_files_in_directory(directory_path: str) -> int:
    try:
        return sum(
            1 for entry in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, entry))
        )
    except FileNotFoundError:
        print(f"Path not exists: {directory_path}")
        return 0
    except PermissionError:
        print(f"Permission denied: {directory_path}")
        return 0

def judge_4o_all_multi(ref_comments, gen_comment, target_method, random=False):
    prompt = f"""
Task: Match each generated step with its functionally equivalent reference step(s).

Input:
1. Target Method: Scientific research method described in LaTeX.
2. Reference Steps: Ordered steps from the original implementation, labeled [Ref 1], [Ref 2], etc.
3. Generated Steps: Ordered steps from an LLM-generated implementation, labeled [Gen 1], [Gen 2], etc.

Output Requirements:
1. For each generated step, identify the reference step(s) that implement the same specific functionality.
2. Format your answer as follows:
   Gen 1: Ref X
   Gen 2: Ref Y, Ref Z
   Gen 3: -1
   
3. Matching criteria:
   - Match based on functional equivalence, not textual similarity
   - Steps must perform the same specific operation
   - Steps must serve the same role in the overall algorithm
   - Steps must produce equivalent results given the same inputs
   
4. Consider sequential position:
   - Earlier generated steps likely match earlier reference steps
   - Later generated steps likely match later reference steps
   
5. If a generated step has no clear equivalent or is ambiguous, output "-1"

6. Important:
   - Ensure all reference indices actually exist in the reference steps
   - Do not include explanations in your output
   - Provide answers for all generated steps

Input Format:

[Target Method]
{target_method}

[Reference Steps]
{ref_comments}

[Generated Steps]
{gen_comment}

Your answer:
"""
    response = llm(prompt, model='gpt-4o', stop=None)
    return response

def merge_nodes(dependencies, node_match_list):
    """
    Merges nodes in a dependency graph that match the same reference node.

    Args:
      dependencies: A dictionary representing the dependencies in the graph.
        Keys are snippet names (strings), and values are lists of dependencies (strings).
      node_match_list: A list indicating the mapping between nodes in the reasoning graph
        and the reference graph. The index represents the node in the reasoning graph
        (starting from 0), and the value represents the corresponding node's index in
        the reference graph (starting from 0).

    Returns:
      A new dictionary representing the merged dependencies.
    """
    merged_dependencies = {}
    node_map = {}  # Map from original node name to merged node name

    # Create a mapping of which nodes to merge
    for i, ref_index in enumerate(node_match_list):
        if ref_index != -1:
            reasoning_node_name = f"Snippet {i + 1}"
            reference_node_name = f"Snippet {ref_index + 1}"
            
            if reference_node_name not in node_map:
                node_map[reference_node_name] = []
            
            node_map[reference_node_name].append(reasoning_node_name)


    # Merge dependencies
    for ref_node, original_nodes in node_map.items():
      merged_dependencies[ref_node] = []
      for original_node in original_nodes:
        if original_node in dependencies:
          merged_dependencies[ref_node].extend(dependencies[original_node])

    # Update dependencies to point to merged nodes
    for ref_node, deps in merged_dependencies.items():
        new_deps = []
        for dep in deps:
            
            found_match = False
            for ref_node_key, original_nodes in node_map.items():
              if dep in original_nodes:
                new_deps.append(ref_node_key)
                found_match = True
                break
            if not found_match:
              new_deps.append(dep)
              
        merged_dependencies[ref_node] = list(set(new_deps))  # Remove duplicates
    
    #Remove dependencies that points to node itself
    for ref_node, deps in merged_dependencies.items():
      merged_dependencies[ref_node] = [d for d in deps if d!=ref_node]

    return merged_dependencies

def calculate_edge_scores(dependencies, node_scores):
    """
    Calculates the score for each edge in a graph, normalizing the scores
    so that the total score of all edges is 1.

    Args:
      dependencies: A dictionary mapping node names to a list of their dependencies.
      node_scores: A dictionary mapping node names to their scores.

    Returns:
      A dictionary mapping edges (tuples of node names) to their normalized scores.
    """
    edge_scores = {}
    for node, deps in dependencies.items():
        for dep in deps:
            edge_scores[(dep, node)] = node_scores.get(node, 0) * node_scores.get(dep, 0)

    # Calculate the total score of all edges.
    total_score = sum(edge_scores.values())

    # Avoid division by zero if there are no edges.
    if total_score == 0:
        return edge_scores

    # Normalize the edge scores.
    normalized_edge_scores = {
        edge: score / total_score for edge, score in edge_scores.items()
    }

    return normalized_edge_scores

def calculate_matched_edge_score(reference_dependencies, ref_node_scores, matched_edge):
    reference_edge_scores = calculate_edge_scores(reference_dependencies, ref_node_scores)
    scores = 0
    if len(reference_edge_scores) == 0:
        return 1  # No edges in reference, so no edge score
    for i in range(len(matched_edge)):
        ref_dep, ref_node = matched_edge[i]
        if (ref_dep, ref_node) in reference_edge_scores:
            scores += reference_edge_scores[(ref_dep, ref_node)]
    return scores

def edge_match_multi(reference_graph, target_graph, match_list):

    def find_matched_edges(reference_adj, target_adj, ref_to_targets):
        matched_edges = []

        # The reference_adj is an adjacency list where reference_adj[a] = list of nodes b such that there's an edge a->b.
        # We'll iterate over each reference node a.
        for a in range(len(reference_adj)):
            for b in reference_adj[a]:
                # This means we have an edge a -> b in the reference.
                ref_edge_name = f"{index_to_snippet_name(a)} -> {index_to_snippet_name(b)}"
                print(f"Checking reference edge: {ref_edge_name}")
                
                # We'll check if there's a path in the target from any target node that matches a to any target node that matches b.
                if a not in ref_to_targets or b not in ref_to_targets:
                    print(f"  Skipping - missing matches: a={a in ref_to_targets}, b={b in ref_to_targets}")
                    continue  # If we don't have any target match for either node, skip

                # Check all pairs of target nodes
                edge_found = False
                print(f"  Checking matches: a -> {ref_to_targets[a]}, b -> {ref_to_targets[b]}")

                for m in range(len(ref_to_targets[a])):
                    for n in range(len(ref_to_targets[b])):
                        a_target = ref_to_targets[a][m]
                        b_target = ref_to_targets[b][n]
                        
                        print(f"    Testing path: {a_target} -> {b_target}")
                        if exists_path_in_target(a_target, b_target, target_adj):
                            print(f"    ✓ Path found!")
                            edge_found = True
                        else:
                            print(f"    ✗ No path")

                        if edge_found:
                            break
                    if edge_found:
                            break

                if edge_found:
                    # Convert indices back to snippet names.
                    edge_tuple = (index_to_snippet_name(a), index_to_snippet_name(b))
                    print(f"  ✓ Edge matched: {edge_tuple}")
                    if edge_tuple not in matched_edges:
                        matched_edges.append(edge_tuple)
                else:
                    print(f"  ✗ Edge not matched: {ref_edge_name}")

        return matched_edges

    # Convert a snippet name (e.g., "Snippet 3") to an integer index (2)
    def snippet_name_to_index(name: str, name_to_idx_map) -> int:
        return name_to_idx_map[name]

    # Convert an integer index (2) back to snippet name ("Snippet 3")
    def index_to_snippet_name(idx: int) -> str:
        return f"Snippet {idx + 1}"

    # Build adjacency lists in terms of indices:
    reference_adj = [[] for _ in range(len(reference_graph))]
    ref_name_to_idx = dict()
    num = 0
    for n in reference_graph.keys():
        ref_name_to_idx[n] = num
        num += 1

    for node, deps in reference_graph.items():
        node_idx = snippet_name_to_index(node, ref_name_to_idx)
        for dep in deps:
            dep_idx = snippet_name_to_index(dep, ref_name_to_idx)
            # Edge dep -> node, so add node_idx to adjacency of dep_idx
            reference_adj[dep_idx].append(node_idx)

    target_adj = [[] for _ in range(len(target_graph))]
    target_name_to_idx = dict()
    num = 0
    for n in target_graph.keys():
        target_name_to_idx[n] = num
        num += 1
    for node, deps in target_graph.items():
        node_idx = snippet_name_to_index(node, target_name_to_idx)
        for dep in deps:
            dep_idx = snippet_name_to_index(dep, target_name_to_idx)
            target_adj[dep_idx].append(node_idx)


    from collections import defaultdict
    ref_to_targets = defaultdict(list)

    # Build mapping from reference indices to target indices
    for t_idx, ref_matches in enumerate(match_list):
        for ref_idx in ref_matches:
            if ref_idx >= 0:  # Valid reference match
                ref_to_targets[ref_idx].append(t_idx)
    
    print("Reference to target mapping:", dict(ref_to_targets))
    print("Reference adjacency:", reference_adj)
    print("Target adjacency:", target_adj)

    # We'll define a BFS function to check if there's a path in the target graph from start to end.
    def exists_path_in_target(start_idx: int, end_idx: int, adjacency) -> bool:
        if start_idx == end_idx:
            return False  # Self-loops should not count as valid edges
        
        visited = set()
        queue = deque([start_idx])
        visited.add(start_idx)

        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt == end_idx:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    # Now, for each edge in the reference graph, we see if there's a path in the target graph.
    # An edge in the reference graph is (dep_idx -> node_idx), meaning node_idx depends on dep_idx.
    # We'll collect the edges that have a match.
    return find_matched_edges(reference_adj, target_adj, ref_to_targets)

def calculate_reason_process_acc_multi(Latex_code, code_ref, code_gen, Num_Global=None, model=None, tokenizer=None):
    analyzer_ref = SnippetDependencyAnalyzer(code_ref)
    dependencies_ref = analyzer_ref.run_analysis()

    delete_key = list()
    for snip_id, snippet_word in analyzer_ref.snippet_comments.items():
        if snip_id not in analyzer_ref.snippet_boundaries.keys():
            delete_key.append(snip_id)
        snippet_word = extract_comments_to_single_line(snippet_word)
        analyzer_ref.snippet_comments[snip_id] = snippet_word
    for key in delete_key:
        del analyzer_ref.snippet_comments[key]

    analyzer_gen = SnippetDependencyAnalyzer(code_gen)
    dependencies_gen = analyzer_gen.run_analysis()

    delete_key = list()
    for snip_id, snippet_word in analyzer_gen.snippet_comments.items():
        if snip_id not in analyzer_gen.snippet_boundaries.keys():
            delete_key.append(snip_id)
            
        snippet_word = extract_comments_to_single_line(snippet_word)
        analyzer_gen.snippet_comments[snip_id] = snippet_word
    for key in delete_key:
        del analyzer_gen.snippet_comments[key]

    total_score = 0
    reference_score = dict()
    for i in analyzer_ref.snippet_comment_code_complexity.keys():
        total_score += analyzer_ref.snippet_comment_code_complexity[i]['total']
    
    if total_score == 0:
        return 0
        
    for i in analyzer_ref.snippet_comment_code_complexity.keys():
        reference_score[i] = analyzer_ref.snippet_comment_code_complexity[i]['total']/total_score

    ref_string = ""
    num = 0
    for snip_id, snippet_word in analyzer_ref.snippet_comments.items():
        ref_string += "Ref " +str(num+1) + ". " + snippet_word.strip() + "\n"
        num += 1

    num = 0
    gen_string = ""
    for snip_id, snippet_word in analyzer_gen.snippet_comments.items():
        gen_string += "Gen " +str(num+1) + ". " + snippet_word.strip() + "\n"
        num += 1
    
    if gen_string == "":
        return 0
    
    # Check for empty graphs
    if len(analyzer_ref.snippet_boundaries) == 0 or len(analyzer_gen.snippet_boundaries) == 0:
        return 0
    
    response = judge_4o_all_multi(ref_string, gen_string, Latex_code, random=False)
    print("GPT-4o Response:\n", response)
    
    # Parse the response more carefully
    lines = response.split('\n')
    match_list = []
    
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        # Extract the part after ':'
        match_part = line.split(':', 1)[-1].strip()
        
        # Split by comma and parse each reference
        ref_matches = []
        if match_part.strip() == "-1":
            ref_matches = [-1]
        else:
            for ref_str in match_part.split(','):
                ref_str = ref_str.strip()
                if not ref_str:
                    continue
                    
                # Extract number from "Ref X" format
                if ref_str.lower().startswith('ref'):
                    ref_num_str = ref_str.split()[-1]
                else:
                    ref_num_str = ref_str
                
                try:
                    ref_num = int(ref_num_str)
                    if ref_num == -1:
                        ref_matches.append(-1)
                    else:
                        ref_matches.append(ref_num - 1)  # Convert to 0-based index
                except ValueError:
                    ref_matches.append(-1)
        
        if ref_matches:  # Only add non-empty matches
            match_list.append(ref_matches)
    
    # Check for empty match list
    if not match_list:
        return 0
    
    # print("Parsed match_list:", match_list)
    
    matched_edge = edge_match_multi(dependencies_ref, dependencies_gen, match_list)
    
    # Calculate matched node scores - sum scores of all matched reference nodes
    matched_ref_nodes = set()  # Use set to avoid double counting
    for gen_idx in range(len(match_list)):
        for ref_idx in match_list[gen_idx]:
            if ref_idx != -1:
                ref_node_name = f"Snippet {ref_idx + 1}"
                matched_ref_nodes.add(ref_node_name)
    
    print("Reference dependencies:", dependencies_ref)
    print("Generated dependencies:", dependencies_gen)
    print("Reference node scores:", reference_score)
    print("Matched reference nodes:", matched_ref_nodes)
    print("Matched edges:", matched_edge)
    
    scores = calculate_matched_edge_score(dependencies_ref, reference_score, matched_edge)
    
    node_scores = 0
    for ref_node_name in matched_ref_nodes:
        if ref_node_name in reference_score:
            node_scores += reference_score[ref_node_name]

    # Ensure both scores are between 0 and 1
    node_scores = min(node_scores, 1.0)
    scores = min(scores, 1.0)
        
    final_score = (scores + node_scores) / 2

    return final_score
