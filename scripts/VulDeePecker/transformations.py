import random
import string
import re

def no_transformation(code):
    
    return code

def tf_1(code):
    
    def rename_parameter(code, old_parameter_name):

        letters = string.ascii_lowercase
        new_parameter_name = ''.join(random.choice(letters) for i in range(2))
        
        parameter = old_parameter_name.replace("*", "")
        parameter = parameter.replace("[", "")
        parameter = parameter.replace("\\", "")
        parameter = parameter.replace("\\", "")
        parameter = parameter.replace("\\P", "")
        
        if parameter not in ["...", "private", ""]:
        
            neutral_characters = ["(", ")", ",", ";", " ", "*", "[", "]", "-", ">", "&", ":"]
            
            occurences = [m.start() for m in re.finditer(parameter, code)]
            num_inserted_chars = 0
            
            for occurence in occurences:
                
                occurence += num_inserted_chars
                
                if occurence + len(parameter) < len(code): 
                
                    prev_char = code[occurence - 1]
                    next_char = code[occurence + len(parameter)]
                    
                    if (prev_char in neutral_characters and next_char in neutral_characters):
                        code = code[0:occurence] + new_parameter_name + code[occurence + len(parameter):]
                        num_inserted_chars += len(new_parameter_name) - len(parameter)               
        
        return code

    if "(" in code and ")" in code:
        parameters = code.split(")")[0].split("(")[1]
        
        if len(parameters) > 0:
            if "," in parameters:
                for param in parameters.split(","):
                    parameter = param.split(" ")[-1]
                    code = rename_parameter(code, parameter)
            else:
                parameter = parameters.split(" ")[-1]
                code = rename_parameter(code, parameter)
            
    return code

import random

def tf_2(code):

    if "(" in code and ")" in code:
        parameters = code.split(")")[0].split("(")[1]
        
        if len(parameters) > 0:
            if "," in parameters:
                parameters = parameters.split(",")
                random.shuffle(parameters)
                new_parameters = ""
                for param in parameters:
                    new_parameters += param.strip()
                    new_parameters += ", "
                new_parameters = new_parameters[:-2]
                
                code = code.split("(")[0] + "(" + new_parameters + ")" + code.split(")")[1]
            
    return code

def tf_3(code):

    letters = string.ascii_lowercase
    new_function_name = ''.join(random.choice(letters) for i in range(2))
    
    if "(" in code and ")" in code:
        before_function = code.split("(")[0]
        
        if " " in before_function:
            function_name = before_function.split(" ")[-1]
            
            if function_name != "" and function_name != " ":
            
                code = code.replace(function_name, new_function_name)
            
    return code


def tf_4(code):
    
    text_to_insert = 'void helpfunc() {\n while (false) {'
        
    for i in range(300):
        text_to_insert += "break;\n"
        
    text_to_insert += ' } \n } \n'

    code = code + "\n\n" + text_to_insert
    
    return code

def tf_5(code):
    
    text_to_insert = '/*'
        
    for i in range(50):
        text_to_insert += "break; "
        
    text_to_insert += '*/'

    code = code + "\n\n" + text_to_insert
    
    return code

def tf_6(code):
    
    placeholder = "placeholderasdfasfd"
    helper_function_name = "helper_func"
    
    if '{' in code:
        begin_of_function = code.index('{')
        if "{" in code and "}" in code:
            end_of_function = code.rindex('}')
            
            occurences = [m.start() for m in re.finditer("\\n", code)]
            
            start_of_function_body = -1
            end_of_function_body = -1
            
            if "(" in code and ")" in code:
                before_function = code.split("(")[0]
                parameters = code.split(")")[0].split("(")[1]
                
                new_params = ""
                
                if len(parameters) > 0:
                    if "," in parameters:
                        for param in parameters.split(","):
                            parameter = param.split(" ")[-1]

                            new_params += parameter.replace("*", "")
                            new_params += ","
                        new_params = new_params[:-1]
                    else:
                        parameter = parameters.split(" ")[-1]
                        new_params = parameter.replace("*", "")
                
                if " " in before_function:
                    function_name = before_function.split(" ")[-1]
            
                    for occurence in occurences:
                        if occurence > begin_of_function and start_of_function_body == -1:
                            start_of_function_body = occurence
                        if occurence < end_of_function:
                            end_of_function_body = occurence
                            
                    if function_name != "" and function_name != " ":
                        
                        function_body = code[start_of_function_body:end_of_function_body+1]
                
                        helper_function = code.replace(function_name, helper_function_name)
                        code_without_function_body = code.replace(function_body, placeholder)
                        main_function = code_without_function_body.replace(placeholder, "\n        return " + helper_function_name + "(" + new_params + ");\n")
                        
                        code = helper_function + "\n\n" + main_function
                
    
    return code

def tf_7(code):
    
    if '{' in code:
        begin_of_function = code.index('{')

        text_to_insert = '                                                                                                                                                                                 '

        code = code[0:begin_of_function + 1] + text_to_insert + code[begin_of_function + 1:]
    
    return code

def tf_8(code):
    
    if '{' in code:
        begin_of_function = code.index('{')

        text_to_insert = '\n    help_func();'

        code = code[0:begin_of_function + 1] + text_to_insert + code[begin_of_function + 1:]
        
        func_to_insert = 'void helpfunc() {\n'
            
        for i in range(150):
            func_to_insert += "return;\n"
            
        func_to_insert += ' } \n'

        code = code + "\n\n" + func_to_insert
    
    return code

def tf_9(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    return(code)

def tf_10(code, training_sample_code):

    code = code + "\n/*" + training_sample_code + "*/"
    
    return code

def tf_11(code, transformations, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply = None):
    
    selected_transformation = random.choice(transformations)
    
    if selected_transformation.__name__ == "tf_10":
        code = selected_transformation(code, training_set_sample_neg)
    elif selected_transformation.__name__ == "tf_13":
        code = selected_transformation(code, training_set_sample_pos)
    elif selected_transformation.__name__ == "tf_11":
        return tf_11(code, transformations, training_set_sample_neg, training_set_sample_pos)
    elif trafo_not_to_apply is not None and selected_transformation.__name__ == trafo_not_to_apply.__name__ :
        return tf_11(code, transformations, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply)
    else:
        code = selected_transformation(code)
    
    return code

def tf_12(code):
    code = re.sub('\n','',code)
    code = re.sub('\t','',code)
    return(code)

def tf_13(code, training_sample_code):

    code = code + "\n/*" + training_sample_code + "*/"
    
    return code
        
            
        
    