def letter_dot(letters, weight):
    res = []
    # weight = np.around(weight, 1)
    # weight = round_to_sigfigs(weight, 3)
    #print(letters, weight)
    assert(len(letters) == weight.shape[1])
    for w in weight:
        ans = []
        for i in range(weight.shape[1]):
            if w[i] == 0:
                continue
            elif letters[i] == '0':
                continue
            elif letters[i] == '1':
                ans.append(str(w[i]))
                continue
            elif '+' in letters[i]:
                ans.append(str(w[i]) + '*' + '(' + letters[i] + ')')
                continue
            elif '-' in letters[i]:
                ans.append(str(w[i]) + '*' + '(' + letters[i] + ')')
                continue
            ans.append(str(w[i]) + '*' + letters[i])
        if not ans:
            ans.append('0')
        cache = ans[0]
        for x in ans[1:]:
            if x[0] == '-':
                cache += x
            else:
                cache += '+' + x
#         res.append('+'.join(ans))
        res.append(cache)
    return res


# def letter_dot_format(letters, weight):
#     res = []
#     # weight = np.around(weight, 1)
#     # weight = round_to_sigfigs(weight, 3)
#     #print(letters, weight)
#     assert(len(letters) == weight.shape[1])
#     for w in weight:
#         ans = []
#         for i in range(weight.shape[1]):
#             if w[i] == 0:
#                 continue
#             elif letters[i] == '0':
#                 continue
#             elif letters[i] == '1':
#                 ans.append("{:.8f}".format(w[i]))
#                 continue
#             elif '+' in letters[i]:
#                 ans.append("{:.8f}".format(w[i]) + '*' + '(' + letters[i] + ')')
#                 continue
#             elif '-' in letters[i]:
#                 ans.append("{:.8f}".format(w[i]) + '*' + '(' + letters[i] + ')')
#                 continue
#             ans.append("{:.8f}".format(w[i]) + '*' + letters[i])
#         if not ans:
#             ans.append('0')
#         cache = ans[0]
#         for x in ans[1:]:
#             if x[0] == '-':
#                 cache += x
#             else:
#                 cache += '+' + x
# #         res.append('+'.join(ans))
#         res.append(cache)
#     return res
def letter_act(letters, layer):
    res = []
    for i,l in enumerate(letters):
        if i == 0:
           res.append(l)
        elif i == 1:
            if l == '0':
                #res.append('1')
                res.append('1')
            else:
                res.append('exp(' + l + ')')
        elif i == 2:
            if l == '0':
                res.append('0')
            else:
                #res.append(l)
                res.append( 'sin(' + l + ')')
        elif i == 3:
            if l == '1':
                #res.append('0')
                res.append('0')
            # elif l == '0':
            #     res.append('log(-' + l + ' +0.01) +2*log(0.005)')
            elif l == '0':
                res.append('-5.3')
            else:
                #res.append('-5.3')
                res.append('ln(' + l + ')')
    return res

def letter_add(letters0, letters1):
    res = []
    assert(len(letters0) == len(letters1))
    for i in range(len(letters0)):
        x = letters0[i]
        y = letters1[i]
        if x == '0':
            res.append(y)
            continue
        elif y == '0':
            res.append(x)
            continue
        else:
            if y[0] == '-':
                res.append(x + y)
            else:
                res.append(x + '+' + y)
    return res




def backtr(letters, model):
    # letters = ['x', '1']
    t = 0
    inputs = [letters]
    add_ons = []
    for i in range(len(model.model)):
        input_x = inputs[-1]
        for j in range(i):
            #print((inputs[j]))

            add_on = letter_dot(inputs[j],model.adjuster[t].weight.clone().detach().numpy())
            #print(add_on)
            input_x = letter_add(input_x, add_on)
           # print(input_x)
            add_ons.append(add_on)
            t += 1
        # if i ==0:
        #     fc_res = letter_dot(input_x, model.cpu().model[i].weight.cpu().clone().detach().numpy())
        # else:
        fc_res = letter_dot(input_x, model.cpu().model[i].fc.weight.cpu().clone().detach().numpy())
        #print(fc_res)
        act_res = letter_act(fc_res,i)
        #print(act_res)
        
        if i!= len(model.model)-1:
            bridge_res = []
            for n in range(4):
                temp = letter_dot([act_res[n]], model.cpu().bridges[i].node[n].weight.cpu().clone().detach().numpy())
                bridge_res.append(temp[0])
            #print(bridge_res)
            inputs.append(bridge_res)
        else:
            # print(act_res[0],act_res)
            act_res[0] = letter_add(act_res, list(map(str, (model.model[i].fc.bias.cpu().clone().detach().numpy())))) #act_res[0] + "+" + str([0])
            #print(act_res)
            inputs.append(act_res)
    return inputs