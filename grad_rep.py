import re
import torch
from admm import cal_sparsity
from sympy import symbols, diff, parse_expr, simplify, log, exp, N, sympify, nsimplify, sin, cos, asin, pi
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from decimal import Decimal
import re
def GradientPruning(model, inputs, pcen_bool = True, thresh = 0.01):
    # print(inputs)
    outputs = model(inputs)  # Forward pass
    gradient_tensor = torch.ones_like(outputs) #The gradient of the function being differentiated w.r.t. self, implying summing gradients
    
    outputs_to_grad = outputs  # Pick the specific output or use the entire output tensor
    outputs_to_grad.backward(gradient = gradient_tensor, retain_graph=True)  # Compute the gradient and retain the graph
    
    # Now you can access the gradients for each parameter (weight)

        
    for module in model.model:
        #print(module)
        if hasattr(module.fc, 'weight'):  
            #grads.append(module.fc.weight.grad)
            #res.append(torch.mul(module.fc.weight.grad, module.fc.weight))
            res = torch.mul(module.fc.weight.grad, module.fc.weight)
            pcen =cal_sparsity(module)
            if pcen_bool:
                threshold = torch.quantile(res.abs(), pcen)
            else:
                threshold = thresh
            mask = res.abs() < threshold  # Identify low-sen sitivity weights
            #print(mask)
            with torch.no_grad():
                module.fc.weight[mask] = 0  # Prune (set to zero) the low-sensitivity weights
               # module.fc.weight.grad[mask] = 0 
            
    # pcen =cal_sparsity(model.adjuster)
        
    for module in model.adjuster:
        #print(module)
        if hasattr(module, 'weight'):  
            #grads.append(module.fc.weight.grad)
            #res.append(torch.mul(module.fc.weight.grad, module.fc.weight))
            res = torch.mul(module.weight.grad, module.weight)
            pcen =cal_sparsity(module)
            if pcen_bool:
                threshold = torch.quantile(res.abs(), pcen)
            else:
                threshold = thresh
            mask = res.abs() < threshold  # Identify low-sensitivity weights
            with torch.no_grad():
                module.weight[mask] = 0  # Prune (set to zero) the low-sensitivity weights
                #module.weight.grad[mask] = 0 
    
    
        
    for module in model.bridges:
        for k in range(len(model.bridges[0].node)):
            sensitivities = []        
            if hasattr(module.node[k], 'weight'):  
                res = torch.mul(module.node[k].weight.grad, module.node[k].weight)
                sensitivities.append(res.abs())
                pcen =cal_sparsity(module)
            sensitivities = torch.cat(sensitivities)
        if pcen_bool:
                threshold = torch.quantile(sensitivities, pcen)
        else:
            threshold = thresh
        
        for k in range(len(model.bridges[0].node)):
            if hasattr(module.node[k], 'weight'):  
                res = torch.mul(module.node[k].weight.grad, module.node[k].weight)       
                mask = res.abs() < threshold  # Identify low-sensitivity weights
                with torch.no_grad():
                    module.node[k].weight[mask] = 0  # Prune (set to zero) the low-sensitivity weights
                    #module.node[k].weight.grad[mask] = 0 

    return model
        
    



def add_multiplication_symbols(expression):
    # List of functions to avoid inserting a * after
    functions = ["log", "exp", "sin", "cos", "asin", "pi"]
    
    # Create a pattern to match functions
    functions_pattern = "|".join(functions)
    
    # Insert * between a number/variable and a non-function (e.g., 2.18182 e^ but not log(S))
    expression = re.sub(rf'(\d+\.?\d*|\b(?!{functions_pattern})\w+\b)(\s*)([a-zA-Z(])', r'\1 * \3', expression)
    
    # Insert * between closing parenthesis and a variable/non-function (e.g., )e^ or )T but not )log)
    expression = re.sub(rf'(\))(\s*)([a-zA-Z(])(?!{functions_pattern})', r'\1 * \3', expression)
    
    # Insert * between variables (e.g., T S -> T * S)
    expression = re.sub(r'(\b\w+\b)(\s+)(\b\w+\b)', r'\1 * \3', expression)
    
    return expression

def replace_numbers_with_vars(expr_str, threshold):
    replacements = {}
    def replace(match):
        num = float(match.group())
        if num < threshold:
            var = symbols(f'var{replace.counter}')
            replacements[str(var)] = num
            replace.counter += 1
            return str(var)
        else:
            return match.group()

    replace.counter = 1
    pattern = re.compile(r'\b\d+\.?\d*\b')
    return pattern.sub(replace, expr_str), replacements

def replace_numbers_with_vars_sympy(expr, threshold):
    replacements = {}
    
    def replace_num(expr):
        # Handle numeric constants below the threshold
        if isinstance(expr, (sp.Float, sp.Integer)) and abs(expr.evalf()) < threshold:
            var = symbols(f'var{replace_num.counter}')
            replacements[var] = expr
            replace_num.counter += 1
            return var
        # Recursively handle expression arguments
        elif expr.args:  # If the expression has arguments, recursively process them
            return expr.func(*[replace_num(arg) for arg in expr.args])
        else:
            return expr  # Return the expression unchanged if not a number or expression with args

    replace_num.counter = 1
    replaced_expr = replace_num(expr)
    
    return replaced_expr, replacements
    
def grad_check(res, *symbols_list, threshold=0.05,tolerance=0.05):
    # res = res.replace('^', '**')
    # res = res.replace('ln', 'log')
    #print(res)
    # res = add_multiplication_symbols(res)
    # print(res)
    
    # Replace numbers below the threshold with variables
    replaced_expr_str, p = replace_numbers_with_vars_sympy(res, threshold)
    if not p:
        return res
    #print(replaced_expr_str)
    e = exp(1)
    #print("Replaced Expression:", replaced_expr_str, p)
    # T, S = symbols('T S')
    symbols_dict = {symbol: symbols(symbol) for symbol in symbols_list}
    #print(symbols_dict)
    print(p)
    p = {str(k): v for k, v in p.items()}
    # Ensure vars_to_diff is always a tuple or list
    if len(p) == 1:
        # If there's only one replacement, create a tuple with one element
        vars_to_diff = (symbols(next(iter(p.keys()))),)
    else:
        # Otherwise, create symbols for each replacement variable
        vars_to_diff = symbols(' '.join(p.keys()))
    
    local_dict = {'log': log, 'exp': exp, 'sin':sin, 'cos':cos, 'arcsin':asin, 'pi':pi}
    local_dict.update(symbols_dict)
    local_dict.update(**{var: symbols(var) for var in p.keys()})
    expr = replaced_expr_str#parse_expr(replaced_expr_str, evaluate=False, local_dict=local_dict)
    gradients = {}
    for var in vars_to_diff:
        gradients[var] = diff(expr, var)
    
    # Simplify the gradients for clarity
    gradients = {var: simplify(grad) for var, grad in gradients.items()}
    
    # Substitute back the original values for the variables
    gradients_substituted = {var: grad.subs(p) for var, grad in gradients.items()}

    is_close_to_zero = {}
    print(gradients_substituted)
    for var, grad in gradients_substituted.items():
        #print(f"Gradient with respect to {var}:", grad)
        evaluated_grad = N(grad)
        midpoint_value = evaluated_grad.subs({v: 1 for v in symbols_dict.values()}).simplify()#evaluated_grad.subs({T: 2, S: 2}).simplify()
        expr2 = sympify(midpoint_value)
        expr2 = expr2.subs(sp.Symbol('e'), sp.E)
        midpoint_value = simplify(expr2)
        midpoint_value = midpoint_value.evalf()
        print(midpoint_value)
        is_close_to_zero[var] = abs(midpoint_value) < tolerance
    
    
    # print(is_close_to_zero)
    # Substitute zero into the original expression if the gradient is close to zero
    modified_expr = expr
    for var, is_zero in is_close_to_zero.items():
        if is_zero:
            modified_expr = modified_expr.subs(var, 0)
        else:
            modified_expr = modified_expr.subs(var, p[str(var)])
    
    return modified_expr 


def is_symbolic_solution(phi_true, phi_hat):
    """
    Check if the learned model phi_hat is a symbolic solution to the ground-truth model phi_true.

    Parameters:
    - phi_true: sympy expression representing the ground-truth model φ*.
    - phi_hat: sympy expression representing the learned model φ̂.

    Returns:
    - True if φ̂ is a symbolic solution, False otherwise.
    """
    # Simplify expressions
    phi_true = sp.simplify(phi_true)
    # phi_hat = sp.simplify(phi_hat)
    phi_hat = (phi_hat).evalf()

    # Check if phi_hat is not a constant
    if phi_hat.is_constant():
        return False

    # Define symbolic variable(s)
    variables = list(phi_true.free_symbols.union(phi_hat.free_symbols))

    # Check if (phi_true - phi_hat) is a constant
    difference = sp.simplify(phi_true - phi_hat)
    if difference.is_constant():
        return True

    # Check if (phi_true / phi_hat) is a constant and not zero
    ratio = sp.simplify(phi_true / phi_hat)
    if ratio.is_constant() and ratio != 0:
        return True

    # If neither condition is met, return False
    return False




def custom_round(value, precision=5):
    """
    Rounds a value to avoid floating-point precision issues, by using nsimplify.
    """
    return value.evalf(precision)#nsimplify(value, tolerance=10**(-precision))

def grad_round_check(expr, *symbols_list,data, threshold=0.01, rounding_tolerance=1):
    # Convert the input symbols list to sympy symbols
    symbols_dict = {symbol: symbols(symbol) for symbol in symbols_list}
    
    replaced_expr_str, p = replace_numbers_with_vars_sympy(expr, threshold)
    # print(replaced_expr_str)
    # print(p)
    # Extract constants from the expression that need to be checked
    constants = [s for s in replaced_expr_str.free_symbols if str(s) not in symbols_list]

    # print(constants)
    # Calculate gradients with respect to each constant
    gradients = {const: simplify(diff(replaced_expr_str, const)) for const in constants}
    # print(gradients)
    # Check which constants have gradients small enough to justify rounding
    rounding_decisions = {}

    for const, grad in gradients.items():
        print(const)
        nearest_int=round(p[const],1)
        safe_nearest_int = nearest_int if nearest_int != 0 else 1e-10
        dedu=abs(nearest_int-p[const])
        if dedu==0:
            rounded_value=int(p[const])
            rounding_decisions[const] = rounded_value
            continue
        evaluated_grad = N(grad)
        # midpoint_value = midpoint_value.subs({v: x_data_np[12000][i] for i,v in enumerate(symbols_dict.values())}).simplify()
        midpoint_value = evaluated_grad.subs({k: v for k, v in p.items() if k != const}).simplify()
        # midpoint_value = midpoint_value.subs({const: safe_nearest_int}).simplify()
        midpoint_value = midpoint_value.subs({const: safe_nearest_int})
        # midpoint_value = midpoint_value.subs({v: float(data[i]) for i, v in enumerate(symbols_dict.values())}).simplify()
        func = lambdify(list(symbols_dict.values()),midpoint_value, modules="numpy")
        outputs= func(*data.T)

        if not isinstance(outputs, float):
            cleaned_output = outputs[~np.isnan(outputs)]
        else:
            cleaned_output=np.array(outputs, ndmin=1)
        # cleaned_output=cleaned_output
        if cleaned_output.size==0:
            rounded_value = nearest_int
            rounding_decisions[const] = rounded_value
        elif max(abs(cleaned_output))*dedu< rounding_tolerance:
            rounded_value = nearest_int  # Adjust precision if needed
            if rounded_value==int(rounded_value):
                rounding_decisions[const] = int(rounded_value)
            else:
                rounding_decisions[const] = rounded_value
        else:
            nearest_int_inv=round(1/p[const],1)
            safe_nearest_int_inv = nearest_int_inv if nearest_int_inv != 0 else 1e-10
            dedu_inv = abs(nearest_int_inv - 1/p[const])
            midpoint_value_inv = evaluated_grad.subs({const: 1/const}).simplify()
            midpoint_value_inv = midpoint_value_inv.subs({k: v for k, v in p.items() if k != const}).simplify()
            # midpoint_value_inv = midpoint_value_inv.subs({const: safe_nearest_int_inv}).simplify()
            midpoint_value_inv = midpoint_value_inv.subs({const: safe_nearest_int_inv}).evalf()
            func_inv = lambdify(list(symbols_dict.values()), midpoint_value_inv, modules="numpy")
            outputs_inv = func_inv(*data.T)
            cleaned_output_inv = outputs_inv[~np.isnan(outputs_inv)]
            if max(abs(cleaned_output_inv))*dedu_inv < rounding_tolerance :
                rounded_value = 1/nearest_int_inv  # Adjust precision if needed
                if rounded_value==int(rounded_value):
                    rounding_decisions[const] = int(rounded_value)
                else:
                    rounding_decisions[const] = rounded_value
            else:
                rounding_decisions[const] = p[const]
    # Substitute rounded values back into the original expression
    print(rounding_decisions)
    modified_expr = replaced_expr_str
    for const, rounded_val in rounding_decisions.items():
        print(const)
        print(modified_expr)
        modified_expr = modified_expr.subs(const, rounded_val).evalf()
        #.evalf()

    return modified_expr


def replace_scientific_notation(match):
    base, exponent = match.groups()
    # Use float and format to force decimal representation
    return "{:.10f}".format(float(f"{base}e{exponent}"))

def symplifyE(res, *symbols_list):
    res = res.replace('^', '**')
    res = res.replace('ln', 'log')
    scientific_notation_pattern = re.compile(r'(\d+\.?\d*)e([-+]?\d+)')
    res = re.sub(scientific_notation_pattern, replace_scientific_notation, res)
    # res = res.replace("e", "*10**")
    #print(res)
    res = add_multiplication_symbols(res)
    

    e = exp(1)
    #print("Replaced Expression:", replaced_expr_str, p)
    # T, S = symbols('T S')
    symbols_dict = {symbol: symbols(symbol) for symbol in symbols_list}
    #print(symbols_dict)
    
    local_dict = {'log': log, 'exp': exp, 'sin':sin, 'cos':cos, 'arcsin':asin, 'pi':pi}
    local_dict.update(symbols_dict)

    
    # Parse the expression into a sympy expression
    # print(replaced_expr_str)
    # print(local_dict)
    expr = parse_expr(res, evaluate=True, local_dict=local_dict)
    return expr
