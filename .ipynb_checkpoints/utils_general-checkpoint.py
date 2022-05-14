from utils_libs import *
from utils_dataset import *
from utils_models import *
# Global parameters
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst

# --- Helper functions

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl


def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

# --- Train functions

def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, hpo_method, hpo_epoch, hpo_maxiter, with_hpo, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
        
    if with_hpo:
        if hpo_method == 'hpona':
            pass
        elif hpo_method == 'decay':
            # Already implemented with lr_decay_rate as constant per round
            pass
        elif hpo_method == 'nm':
            bnds = opt.Bounds(0, 1)
            prms = [learning_rate]
            hpo_results = opt.minimize(train_model_hpo, prms, args = (model, trn_x, trn_y, batch_size, hpo_epoch, weight_decay, dataset_name), bounds=bnds, method='Nelder-Mead', options={'maxiter':hpo_maxiter})
                    
            # Ignore results when learning rate would be set to zero
            if hpo_results.x[0] > 0:
                learning_rate = hpo_results.x[0]
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for e in range(epoch):
        # Training
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, learning_rate

def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch, print_per, weight_decay, hpo_method, hpo_n_minibatch, hpo_maxiter, with_hpo, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False
    
    if with_hpo:
        if hpo_method == 'hpona':
            pass
        elif hpo_method == 'decay':
            # Already implemented with lr_decay_rate
            pass
        elif hpo_method == 'nm':
            bnds = opt.Bounds(0, 1)
            prms = [learning_rate]
            hpo_results = opt.minimize(train_scaffold_mdl_hpo, prms, args = (model, model_func, state_params_diff, trn_x, trn_y, batch_size, hpo_n_minibatch, weight_decay, dataset_name), bounds=bnds, method='Nelder-Mead', options={'maxiter':hpo_maxiter})        
            
            # Ignore results when learning rate would be set to zero
            if hpo_results.x[0] > 0:
                learning_rate = hpo_results.x[0]
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    step_loss = 0; n_data_step = 0
    for e in range(epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay)/2 * np.sum(params * params)
                print("Step %3d, Training Loss: %.4f" %(count_step, step_loss))
                step_loss = 0; n_data_step = 0
                model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, learning_rate

def train_feddyn_mdl(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, hpo_method, hpo_epoch, hpo_maxiter, with_hpo, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    if with_hpo:
        if hpo_method == 'hpona':
            pass
        elif hpo_method == 'decay':
            # Already implemented with lr_decay_rate
            pass
        elif hpo_method == 'nm':
            bnds = opt.Bounds(0, 1)
            prms = [learning_rate]
            hpo_results = opt.minimize(train_feddyn_mdl_hpo, prms, args = (model, model_func, avg_mdl_param, local_grad_vector, trn_x, trn_y, batch_size, hpo_epoch, weight_decay, dataset_name, alpha_coef), bounds=bnds, method='Nelder-Mead', options={'maxiter':hpo_maxiter})
            
            # Ignore results when learning rate would be set to zero
            if hpo_results.x[0] > 0:
                learning_rate = hpo_results.x[0]
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, learning_rate, alpha_coef

def train_fedprox_mdl(model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, hpo_method, hpo_epoch, hpo_maxiter, with_hpo, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = len(avg_model_param_)
    
    if with_hpo:
        if hpo_method == 'hpona':
            pass
        elif hpo_method == 'decay':
            # Already implemented with lr_decay_rate
            pass
        elif hpo_method == 'nm':
            bnds = opt.Bounds(0,1)
            prms = [learning_rate]
            hpo_results = opt.minimize(train_fedprox_mdl_hpo, prms, args = (model, avg_model_param_, trn_x, trn_y, batch_size, hpo_epoch, weight_decay, dataset_name, mu), bounds=bnds, method='Nelder-Mead', options={'maxiter':hpo_maxiter})
            
            # Ignore results when learning rate would be set to zero
            if hpo_results.x[0] > 0:
                learning_rate = hpo_results.x[0]
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, learning_rate, mu

# --- Hyperparameter Optimization (HPO) Train functions

def train_model_hpo(params, model, trn_x, trn_y, batch_size, hpo_epoch, weight_decay, dataset_name):
    
    ### Training section
    
    pct = 0.25
    
    temp_trn_x, tst_x, temp_trn_y, tst_y = train_test_split(trn_x, trn_y, test_size=pct, random_state=42)
    
    trn_x = temp_trn_x
    trn_y = temp_trn_y
    
    n_trn = trn_x.shape[0]
    
    ### End splitting data, modify tst_gen
    
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # Params and copy model
    learning_rate = params[0]

    model_copy = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model_copy.train(); model_copy = model_copy.to(device)
    
    for e in range(hpo_epoch):
        # Training
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model_copy(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model_copy.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
        
    # Freeze model
    for params in model_copy.parameters():
        params.requires_grad = False
    model_copy.eval()
    
    ### Loss Evaluation Section
    
    loss_overall = 0;
    batch_size = min(6000, trn_x.shape[0])
    n_tst = trn_x.shape[0]
    tst_gen = data.DataLoader(Dataset(tst_x, tst_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model_copy.eval(); model_copy = model_copy.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model_copy(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
    
    loss_overall /= n_tst
    if weight_decay != None:
        # Add L2 loss
        params = get_mdl_params([model_copy], n_par=None)
        loss_overall += weight_decay/2 * np.sum(params * params)
        
    model_copy.train()
    return loss_overall

def train_scaffold_mdl_hpo(params, model, model_func, state_params_diff, trn_x, trn_y, batch_size, hpo_n_minibatch, weight_decay, dataset_name):
    
    ### Training Section
    
    pct = 0.25
    
    temp_trn_x, tst_x, temp_trn_y, tst_y = train_test_split(trn_x, trn_y, test_size=pct, random_state=42)
    
    trn_x = temp_trn_x
    trn_y = temp_trn_y
    
    ### End splitting data, modify tst_gen
    
    n_trn = trn_x.shape[0]
    
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        
    # Params and copy model
    model_copy = copy.deepcopy(model)
    learning_rate = params[0]
    
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model_copy.train(); model_copy = model_copy.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
    hpo_epoch = np.ceil(hpo_n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False
    
    step_loss = 0; n_data_step = 0
    for e in range(hpo_epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            count_step += 1
            if count_step > hpo_n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model_copy(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model_copy.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model_copy.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]
    
    # Freeze model
    for params in model_copy.parameters():
        params.requires_grad = False
    model_copy.eval()
    
    ### Loss Evaluation Section
    
    loss_overall = 0;
    batch_size = min(6000, trn_x.shape[0])
    n_tst = trn_x.shape[0]
    tst_gen = data.DataLoader(Dataset(tst_x, tst_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model_copy.eval(); model_copy = model_copy.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model_copy(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
    
    loss_overall /= n_tst
    if weight_decay != None:
        # Add L2 loss
        params = get_mdl_params([model_copy], n_par=None)
        loss_overall += weight_decay/2 * np.sum(params * params)
        
    model_copy.train()
    return loss_overall

def train_feddyn_mdl_hpo(params, model, model_func, avg_mdl_param, local_grad_vector, trn_x, trn_y, batch_size, hpo_epoch, weight_decay, dataset_name, alpha_coef):
        
    ### Training Section
    
    pct = 0.25
    
    temp_trn_x, tst_x, temp_trn_y, tst_y = train_test_split(trn_x, trn_y, test_size=pct, random_state=42)
    
    trn_x = temp_trn_x
    trn_y = temp_trn_y
    
    n_trn = trn_x.shape[0]

    ### End splitting data, modify tst_gen
    
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # Params and copy model
    model_copy = copy.deepcopy(model)
    learning_rate = params[0]
    
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model_copy.train(); model_copy = model_copy.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    for e in range(hpo_epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model_copy(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model_copy.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model_copy.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]
    
    # Freeze model
    for params in model_copy.parameters():
        params.requires_grad = False
    model_copy.eval()
            
    ### Loss Evaluation Section
    
    loss_overall = 0;
    batch_size = min(6000, trn_x.shape[0])
    n_tst = trn_x.shape[0]
    tst_gen = data.DataLoader(Dataset(tst_x, tst_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model_copy.eval(); model_copy = model_copy.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model_copy(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
    
    loss_overall /= n_tst
    if weight_decay != None:
        # Add L2 loss
        params = get_mdl_params([model_copy], n_par=None)
        loss_overall += weight_decay/2 * np.sum(params * params)
        
    model_copy.train()
    return loss_overall

def train_fedprox_mdl_hpo(params, model, avg_model_param_, trn_x, trn_y, batch_size, hpo_epoch, weight_decay, dataset_name, mu):
    
    ### Training Section
    
    pct = 0.25
    
    temp_trn_x, tst_x, temp_trn_y, tst_y = train_test_split(trn_x, trn_y, test_size=pct, random_state=42)
    
    trn_x = temp_trn_x
    trn_y = temp_trn_y
    
    n_trn = trn_x.shape[0]
    
    ### End Splitting Data, modify tst_gen
    
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # Params and copy model
    model_copy = copy.deepcopy(model)
    learning_rate = params[0]
    
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model_copy.train(); model_copy = model_copy.to(device)
    
    n_par = len(avg_model_param_)
    
    for e in range(hpo_epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model_copy(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model_copy.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model_copy.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]
    
    # Freeze model
    for params in model_copy.parameters():
        params.requires_grad = False
    model_copy.eval()
            
    ### Loss Evaluation Section
    
    loss_overall = 0;
    batch_size = min(6000, trn_x.shape[0])
    n_tst = trn_x.shape[0]
    tst_gen = data.DataLoader(Dataset(tst_x, tst_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model_copy.eval(); model_copy = model_copy.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model_copy(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
    
    loss_overall /= n_tst
    if weight_decay != None:
        # Add L2 loss
        params = get_mdl_params([model_copy], n_par=None)
        loss_overall += weight_decay/2 * np.sum(params * params)
        
    model_copy.train()
    return loss_overall