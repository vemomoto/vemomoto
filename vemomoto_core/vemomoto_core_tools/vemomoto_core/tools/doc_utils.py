'''
Created on 31.01.2020

@author: Samuel
'''
import inspect
import re 

IGNORE_STR = "#"
PRIVATE_STR = "~+~"
INSERT_STR = "<!"
APPEND_STR = ">!"

def should_ignore(string):
    return not string or not string.strip() or string.lstrip().startswith(IGNORE_STR)
def should_insert(string):
    return string.lstrip().startswith(INSERT_STR)
def should_append(string):
    return string.lstrip().startswith(APPEND_STR)

class DocMetaSuperclass(type):
    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        if bases:
            for name, member in cls_dict.items():
                for base in bases:
                    if hasattr(base, name):
                        add_parent_doc(member, getattr(bases[-1], name))
                        break
        return cls

def add_doc(*fromfuncs):
    """
    Decorator: Copy the docstring of `fromfunc`
    """
    def _decorator(func):
        for fromfunc in fromfuncs:
            add_parent_doc(func, fromfunc)
        return func
    return _decorator


def strip_private(string:str):
    if PRIVATE_STR not in string:
        return string
    result = ""
    for line in string.splitlines(True):
        if line.strip()[:len(PRIVATE_STR)] == PRIVATE_STR:
            return result
        result += line
    return result

def merge(child_str, parent_str, indent_diff=0, joinstr="\n"):
    parent_str = adjust_indent(parent_str, indent_diff)
    if should_ignore(child_str):
        return parent_str
    if should_append(child_str):
        return joinstr.join([parent_str, re.sub(APPEND_STR, "", child_str, count=1)])
    if should_insert(child_str):
        return joinstr.join([re.sub(INSERT_STR, "", child_str, count=1), parent_str])
    return child_str

def add_parent_doc(child, parent):
    
    if type(parent) == str:
        doc_parent = parent
    else:
        doc_parent = parent.__doc__
    
    if not doc_parent:
        return
    
    doc_child = child.__doc__ if child.__doc__ else ""
    if not callable(child) or not (callable(parent) or type(parent) == str):
        indent_child = get_indent_multi(doc_child)
        indent_parent = get_indent_multi(doc_parent)
        ind_diff = indent_child - indent_parent if doc_child else 0
        
        try:
            child.__doc__ = merge(doc_child, strip_private(doc_parent), ind_diff)
        except AttributeError:
            pass
        return
    
    vars_parent, header_parent, footer_parent, indent_parent = split_variables_numpy(doc_parent, True)
    vars_child, header_child, footer_child, indent_child = split_variables_numpy(doc_child)
    
    
    if doc_child:
        ind_diff = indent_child - indent_parent 
    else: 
        ind_diff = 0
        indent_child = indent_parent
    
    
    header = merge(header_child, header_parent, ind_diff)
    footer = merge(footer_child, footer_parent, ind_diff)
    
    variables = inspect.getfullargspec(child)[0]
    
    varStr = ""
    
    def add_varStr(var, var_type, var_descr, varStr):
        varStr += "".join([adjust_indent(" ".join([var, var_type]), 
                                           indent_child), var_descr])
    
    for var in variables:
        child_var_type, child_var_descr = vars_child.pop(var, [None, None]) 
        parent_var_type, parent_var_descr = vars_parent.pop(var, ["", ""]) 
        var_type = merge(child_var_type, parent_var_type, ind_diff, joinstr=" ")
        var_descr = merge(child_var_descr, parent_var_descr, ind_diff)
        if bool(var_type) and bool(var_descr):
            add_varStr(var, var_type, var_descr, varStr)
    
    for var, (child_var_type, child_var_descr) in vars_child.items():
        parent_var_type, parent_var_descr = vars_parent.pop(child_var_type, ["", ""]) 
        var_type = merge(child_var_type, parent_var_type, ind_diff, joinstr=" ")
        var_descr = merge(child_var_descr, parent_var_descr, ind_diff)
        if bool(var_descr):
            add_varStr(var, var_type, var_descr, varStr)
            
    if varStr.strip():
        varStr = "\n".join([adjust_indent("\nParameters\n----------", 
                                          indent_child), varStr])
    
    child.__doc__ = "\n".join([header, varStr, footer])
    
def adjust_indent(string:str, difference:int) -> str:    
    if not string:
        if difference > 0:
            return " " * difference
        else:
            return ""
    if not difference:
        return string
    if difference > 0:
        diff = " " * difference
        return "".join(diff + line for line in string.splitlines(True))
    else:
        diff = abs(difference)
        result = ""
        for line in string.splitlines(True):
            if get_indent(line) <= diff:
                result += line.lstrip()
            else:
                result += line[diff:]
        return result
    
        
def get_indent(string:str) -> int:
    return len(string) - len(string.lstrip())

def get_indent_multi(string:str) -> int:
    lines = string.splitlines()
    if len(lines) > 1:
        return get_indent(lines[1])
    else:
        return 0

def split_variables_numpy(docstr:str, stripPrivate:bool=False):
    
    if not docstr.strip():
        return {}, docstr, "", 0
    
    lines = docstr.splitlines(True)
    
    header = ""
    for i in range(len(lines)-1):
        if lines[i].strip() == "Parameters" and lines[i+1].strip() == "----------":
            indent = get_indent(lines[i])
            i += 2
            break
        header += lines[i]
    else:
        return {}, docstr, "", get_indent_multi(docstr)
            
    variables = {}
    while i < len(lines)-1 and lines[i].strip():
        splitted = lines[i].split(maxsplit=1)
        var = splitted[0]
        if len(splitted) > 1:
            varType = splitted[1]
        else:
            varType = " "
        varStr = ""
        i += 1
        while i < len(lines) and get_indent(lines[i]) > indent:
            varStr += lines[i]
            i += 1
        if stripPrivate:
            varStr = strip_private(varStr)
        variables[var] = (varType, varStr)
        
    footer = ""
    while i < len(lines):
        footer += lines[i]
        i += 1
    
    if stripPrivate:
        header = strip_private(header)
        footer = strip_private(footer)
    
    return variables, header, footer, indent
    