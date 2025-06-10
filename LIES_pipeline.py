import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
from model import *
from preprocess import *
from backtrack import *
import multiprocessing
import copy
import torch.nn.utils.prune as prune
from admm import *
import os
import nest_asyncio
from sympy import symbols, exp, log, sin, cos, asin, pi,nsimplify
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
from grad_rep import *
import pandas as pd
from sklearn.metrics import r2_score
import re
from decimal import Decimal
#############
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
thres = 0.01
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
noise_level = 0  # Currently not implemented
max_API_attempts = 4
input_options = ["x", "y", "z", "a", "b", "c", "i", "j", "k"]  # These are what we can choose from, doesn't have to be exact same naming
run_name = 'alaki'

data_att = pd.read_csv(r"../BenchData/FeynmanEquations.csv")
eqns = data_att["Filename"].tolist()
eqns = [x for x in eqns if str(x) != 'nan']

eqns = ["II.11.20"]

var_nbs = data_att["# variables"].tolist()
var_nbs = [5]

# for each data eqn
for e, eqn in enumerate(eqns):
    # create dataframe to store results and later save csv (trial#, eqn, r2, spars)
    results = pd.DataFrame(columns=["trial#", "r2", "eqn", "spars", "sym_solu","log-space"])
    n = int(var_nbs[e])

    log_res = []

    dataloader_train1, dataloader_test, min_val1, max_val1, min_vals_output1, max_vals_output1 = dataloader_custom(
        eqn=eqn, logT=True,
        batch_size=128, rate=1,
        norm=True, trim=0.01)

    criterion = ExpLoss()

    dataloader_test1 = copy.deepcopy(dataloader_train1)
    model1 = jump_node_model(inputs=n, hidden_layers=n).to(device)
    optimizer = optim.RMSprop(model1.parameters(), lr=1.5e-2)
    dataset_size = {'train': len(dataloader_train1.dataset),
                    'test': len(dataloader_test1.dataset)}

    flag = 0
    stop = 0

    while flag == 0:
        stop, flag, loss_re, acc_re, r2_re = admm_train(dataloader_train1, dataloader_test1,
                                                        model1, criterion, optimizer, 20, dataset_size,
                                                        l1_lambda=0.05, rho=0.9, admm_thres=0.8,
                                                        pcen_avai=False, pcen=0.75, simple=False, name="Log_",
                                                        alpha=5e-4, l1_update=True, test_skipper=500,
                                                        l1_pcen=False, run_name=run_name, device=device)
        if stop == 1:
            continue

    model_base = jump_node_model(inputs=n, hidden_layers=n).to(device)
    model_base.load_state_dict(torch.load(run_name + 'Log_best_model_l1.pth', weights_only=True))
    dataloader_train = copy.deepcopy(dataloader_train1)
    dataloader_test = copy.deepcopy(dataloader_train)
    min_val = min_val1
    max_val = max_val1
    min_vals_output = min_vals_output1
    max_vals_output = max_vals_output1
    criterion = ExpLoss()
    logT = True

    del log_res, model1, dataloader_train1, dataloader_test1


    ####### Oversampling Start
    error_list = []
    stop, flag, loss_re, acc_re, counts, step_error, score_total = admm_test(dataloader_train, dataloader_test, min_val,
                                                                             max_val, model_base, criterion, optimizer,
                                                                             1,
                                                                             dataset_size, l1_lambda=0.01, rho=0.05,
                                                                             admm_thres=0.5,
                                                                             pcen_avai=True, pcen=0.95, test_only=True,
                                                                             n_input=n, run_name=run_name)

    print(score_total)
    print(cal_sparsity(model_base))
    error_list.append(score_total)
    for step in range(1,2):
        step_error = np.nan_to_num(step_error, nan=np.inf)
        where = np.where(step_error == step_error.min())
        linspaces = []
        upper = []
        lower = []
        for i in range(n):
            if logT:
                bin_edges = torch.linspace(torch.exp(min_val[i]), torch.exp(max_val[i]), 9)
                upper.append(torch.log(bin_edges[int(where[i]) + 1]))
                lower.append(torch.log(bin_edges[int(where[i])]))
            else:
                bin_edges = torch.linspace(min_val[i], max_val[i], 9)
                upper.append(bin_edges[int(where[i]) + 1])
                lower.append(bin_edges[int(where[i])])

        #### Recreate the dataset with desired ranges
        dataloader_train1, dataloader_test1, min_val, max_val, min_vals_output, max_vals_output = dataloader_custom_bias(
            eqn=eqn, logT=logT, batch_size=128, rate=1,
            norm=True, trim=0.1,
            bias=(upper, lower, 0.3))

        dataset_size = {'train': len(dataloader_train1.dataset), 'test': len(dataloader_test1.dataset)}
        model_temp = copy.deepcopy(model_base)
        optimizer = optim.RMSprop(model_temp.parameters(), lr=1.5e-2)

        print("Running weak ADMM for oversampling")
        flag = 0
        stop = 0
        while flag == 0:
            stop, flag, loss_re, acc_re, r2_re = admm_train(dataloader_train1, dataloader_test1,
                                                            model_temp, criterion, optimizer, 20, dataset_size,
                                                            l1_lambda=0.05, rho=0.1, admm_thres=0.8,
                                                            pcen_avai=False, pcen=0.75, simple=False,
                                                            alpha=5e-4, l1_update=True, test_skipper=500,
                                                            l1_pcen=False, run_name=run_name, device=device)
            if stop == 1:
                continue

        stop, flag, loss_re, acc_re, counts, step_error, score_total = admm_test(dataloader_train1, dataloader_test1,
                                                                                 min_val,
                                                                                 max_val, model_temp, criterion,
                                                                                 optimizer,
                                                                                 1,
                                                                                 dataset_size, l1_lambda=0.01, rho=0.05,
                                                                                 admm_thres=0.5,
                                                                                 pcen_avai=True, pcen=0.95,
                                                                                 test_only=True,
                                                                                 n_input=n, run_name=run_name)

        print(score_total)
        print(cal_sparsity(model_temp))
        print('!!!!!!!!!!!!!!!!!!!!!!!!! We are in step{} of oversampling !!!!!!!!!!!!!!!!!!!!!!!!!'.format(step))
        error_list.append(score_total)

        if error_list[step]>error_list[step-1]:
            model_base=model_temp
            dataloader_train=dataloader_train1
            dataloader_test = dataloader_test1
            continue

        elif error_list[step]<error_list[step-1]:
            break
    ##### Oversampling End


    symbols_list = input_options[:n]
    if logT:
        letters = ["ln({})".format(x) for x in symbols_list]
        letters = [f"{s}/{i}" for s, i in zip(letters, max_val)]
    else:
        letters = [x for x in symbols_list]
        letters = [f"{s}/{i}" for s, i in zip(letters, max_val)]

    letters = letters + ['1']

    #### Strong ADMM
    trials = 3
    for j in range(trials):
        try:
            model = copy.deepcopy(model_base)
            optimizer = optim.RMSprop(model.parameters(), lr=1.5e-2)
            print("trial {} running stronger ADMM".format(i))
            flag = 0
            stop = 0
            while flag == 0:
                stop, flag, loss_re, acc_re, r2_re = admm_train(dataloader_train, dataloader_test,
                                                                model, criterion, optimizer, 30, dataset_size,
                                                                l1_lambda=0.01, rho=0.005, admm_thres=0.5,
                                                                pcen_avai=False, pcen=0.95, decay=0.95, simple=False,
                                                                alpha=5 * (10 ** (-n + 1)), l1_update=True,
                                                                test_skipper=500, run_name=run_name, device=device)
                if stop == 1:
                    continue

            print(cal_sparsity(model))
            for data, _ in dataloader_test:
                inputs = data.clone().detach().requires_grad_(True).to(
                    device)
                break

            GradientPruning(model, inputs)

            # stop, flag, loss_re, acc_re, r2_re, counts = admm_train(dataloader_train, dataloader_test,
            #                                                         model, criterion, optimizer, 1, dataset_size,
            #                                                         l1_lambda=0.01, rho=0.05, admm_thres=0.5,
            #                                                         pcen_avai=True, pcen=0.95, test_only=True,
            #                                                         simple=False, run_name=run_name, device=device)
            # print(acc_re)
            # print(counts)
            # print(cal_sparsity(model))


            best_model_wts = copy.deepcopy(model.state_dict())
            if not os.path.exists('./Results/model_res/' + str(eqn)):
                os.makedirs('./Results/model_res/' + str(eqn))
            torch.save(best_model_wts, './Results/model_res/' + str(eqn) + "/trial" + "_" + str(j) + '.pth')
            inputs = backtr(letters, model)

            nest_asyncio.apply()
            a = inputs[-1][0][0]
            print(a)
            attempt = 0
            if logT:
                a = "exp(" + a + ")"

            ##### Simplification
            res = symplifyE(a, *symbols_list)
            dataloader_trainT, dataloader_testT = dataloader_custom(eqn=eqn, logT=False,
                                                                    batch_size=len(dataloader_train.dataset), rate=1,
                                                                    norm=False, trim=0.1)
            x_data_all = []
            y_data_all = []
            count = 0
            for x_batch, y_batch in dataloader_trainT:
                x_data_all.append(x_batch[:, :-1].numpy())
                y_data_all.append(y_batch.numpy())
                count += 1

            x_data_np = (np.vstack(x_data_all))
            y_data_np = np.squeeze(np.vstack(y_data_all))
            del x_data_all, y_data_all
            res2 = grad_round_check(res, *symbols_list, rounding_tolerance=0.1, data=x_data_np,threshold=0.05)
            print(res2)

            symbols_dict = {symbol: sp.symbols(symbol) for symbol in symbols_list}
            res2 = str(res2)

            e = exp(1)
            replaced_expr_str, p = replace_numbers_with_vars(res2, 1e30)
            local_dict = {'log': log, 'e': exp(1), 'sin': sin, 'cos': cos, 'arcsin': asin, 'pi': pi}
            local_dict.update(symbols_dict)
            local_dict.update(**{var: sp.symbols(var) for var in p.keys()})
            print(replaced_expr_str)
            eqnR = sp.sympify(replaced_expr_str, locals=local_dict)
            input_vars = list(symbols_dict.keys())
            input_vars.reverse()
            constants = list(p.keys())

            eqn_func = lambdify(input_vars + constants, eqnR, 'numpy')

            ## Function for regression
            def sympy_equation(x, *params):
                var_inputs = [x[:, i] for i in range(x.shape[1])]
                return eqn_func(*var_inputs, *params)


            def fit_curve():
                try:
                    popt, _ = curve_fit(
                        f=sympy_equation,
                        xdata=x_data_np,
                        ydata=y_data_np,
                        p0=list(p.values()),
                        maxfev=3400
                    )
                    return popt  # Return meaningful parameters
                except RuntimeError:
                    return None

            print("fitting")
            popt = None
            with ThreadPoolExecutor() as executor:
                future = executor.submit(fit_curve)
                try:
                    popt = future.result(timeout=20)
                except TimeoutError:
                    popt = None
                    print('Cant be done')  # Return None if timeout occurs

            ##### Coefficient Optimization
            if popt is not None:
                optimized_constants = dict(zip(constants, popt))
                print(optimized_constants)
                optimized_constants = {key: float(value) for key, value in optimized_constants.items()}
                updated_eqn = eqnR.subs(optimized_constants)
            else:
                updated_eqn = symplifyE(res2)

            updated_eqn1 = grad_round_check(updated_eqn, *symbols_list, rounding_tolerance=0.1,data=x_data_np,threshold=5000)
            updated_eqn1=updated_eqn1.simplify()
            updated_eqn_func = lambdify(input_vars, updated_eqn1, 'numpy')

            var_inputs = [x_data_np[:, i] + 1e-10 for i in range(x_data_np.shape[1])]
            # Predict using the updated equation
            y_pred = updated_eqn_func(*var_inputs)


            if isinstance(y_pred, np.ndarray):
                mean_value = np.nanmean(y_pred)
                y_pred[np.isnan(y_pred)] = mean_value
                r2_re = r2_score(y_data_np, y_pred)
            else:
                y_pred = np.full_like(y_data_np, y_pred)
                r2_re = r2_score(y_data_np, y_pred)


            # fetch GT equation and change it to sympy
            row = data_att[data_att['Filename'] == eqn]
            gt = row['Formula'].iloc[0]
            gt.replace('@', '')
            old_vars = set(re.findall(r'\b[a-zA-Z_]\w*\b', gt))

            # Create a mapping of old variables to new variable names
            variable_mapping = {}
            count = 0
            variable_headers = [col for col in row.columns if col.startswith('v') and 'name' in col]
            variable_names = row[variable_headers].iloc[0].dropna().values.flatten().tolist()
            input_options_temp = input_options[:len(variable_names)]

            mappings = [dict(zip(variable_names, perm)) for perm in permutations(input_options_temp)]

            def replace_variables(formula, mapping):
                # Use regex to replace each variable while ensuring exact matches
                pattern = r'\b({})\b'.format('|'.join(map(re.escape, mapping.keys())))
                return re.sub(pattern, lambda m: mapping[m.group()], formula)
            is_solu=[]
            for i, mapping in enumerate(mappings, 1):
                new_formula = replace_variables(gt, mapping)
                sympy_gt = sp.sympify(new_formula).evalf()

                is_solu.append(is_symbolic_solution(sympy_gt, updated_eqn1))

            new_row = pd.DataFrame(
                [{"trial#": j, "r2": r2_re, "eqn": updated_eqn1, "spars": cal_sparsity(model), "sym_solu": any(is_solu),\
                  "log-space": logT}])
            results = pd.concat([results, new_row], ignore_index=True)
            del model, dataloader_testT, dataloader_trainT
        except:
            print("error")
    results.to_csv("./Results/" + str(eqn) + ".csv", index=False)
