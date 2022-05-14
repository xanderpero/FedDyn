from utils_general import *
from utils_methods import *

# Dataset initialization
###
[alpha, beta, theta, iid_sol, iid_data, name_prefix] = [0.0, 0.0, 0.0, True , True , 'syn_alpha-0_beta-0_theta0']

n_dim = 30
n_clnt= 40
n_cls = 5
avg_data = 200

data_obj = DatasetSynthetic(alpha=alpha, beta=beta, theta=theta, iid_sol=iid_sol, iid_data=iid_data, n_dim=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data, name_prefix=name_prefix)


###
model_name         = 'Linear' # Model type
com_amount         = 400
save_period        = 10
weight_decay       = 1e-5
batch_size         = 10
act_prob           = .1
# lr_decay_per_round = 1      # do not change
epoch              = 10
learning_rate      = 0.01
print_per          = 100
hpo_method         = 'nm' # always use 'nm'
hpo_method_2       = 'hpona' # 'decay' should not be used for synthetic data
hpo_epoch          = 5 # number of steps with hyperparameter configuration
hpo_maxiter        = 5 # number of comparisons with Nelder-Mead
hpo_per            = 20 # perform HPO every X communication rounds

if hpo_method == 'decay':
    lr_decay_per_round = 0.998
else:
    lr_decay_per_round = 1
    
orig_hpo_method = hpo_method
orig_hpo_method = 'nm_.01-10-5-20'
hpo_method_2 = 'hpona_.1'
    
# Model function
model_func = lambda : client_model(model_name, [n_dim, n_cls])
init_model = model_func()

# Initalise the model for all methods
with torch.no_grad():
    init_model.fc.weight = torch.nn.parameter.Parameter(torch.zeros(n_cls,n_dim))
    init_model.fc.bias = torch.nn.parameter.Parameter(torch.zeros(n_cls))

if not os.path.exists('Output/%s/' %(data_obj.name)):
    os.mkdir('Output/%s/' %(data_obj.name))

    
# Methods    
####
# print('FedDyn_%s'%hpo_method)

# alpha_coef = 0.01
# [fed_mdls_sel_FedFyn, trn_perf_sel_FedFyn, tst_perf_sel_FedFyn,
#  fed_mdls_all_FedFyn, trn_perf_all_FedFyn, tst_perf_all_FedFyn,
#  fed_mdls_cld_FedFyn, hyperparameters_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
#                                      learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
#                                      save_period=save_period, lr_decay_per_round=lr_decay_per_round, 
#                                      hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)

# hpo_method = 'hpona'
# print('FedDyn_%s'%hpo_method)
# if hpo_method == 'decay':
#     lr_decay_per_round = 0.998
    
# [fed_mdls_sel_FedFyn_2, trn_perf_sel_FedFyn_2, tst_perf_sel_FedFyn_2,
#  fed_mdls_all_FedFyn_2, trn_perf_all_FedFyn_2, tst_perf_all_FedFyn_2,
#  fed_mdls_cld_FedFyn_2, hyperparameters_FedFyn_2] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
#                                      learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
#                                      save_period=save_period, lr_decay_per_round=lr_decay_per_round, 
#                                      hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)


###
hpo_method=orig_hpo_method

print('SCAFFOLD_%s'%hpo_method)
n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_clnt
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
print_per_ = print_per*n_iter_per_epoch

# calculate hpo_n_minibatch
hpo_n_minibatch = (hpo_epoch*n_iter_per_epoch).astype(np.int64)

[fed_mdls_sel_SCAFFOLD, trn_perf_sel_SCAFFOLD, tst_perf_sel_SCAFFOLD,
 fed_mdls_all_SCAFFOLD, trn_perf_all_SCAFFOLD,
 tst_perf_all_SCAFFOLD, hyperparameters_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                         batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                         print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                         init_model=init_model, save_period=save_period,
                                         lr_decay_per_round=lr_decay_per_round, 
                                         hpo_method=hpo_method, hpo_n_minibatch=hpo_n_minibatch,
                                         hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)

hpo_method = hpo_method_2

print('SCAFFOLD_%s'%hpo_method)

[fed_mdls_sel_SCAFFOLD_2, trn_perf_sel_SCAFFOLD_2, tst_perf_sel_SCAFFOLD_2,
 fed_mdls_all_SCAFFOLD_2, trn_perf_all_SCAFFOLD_2,
 tst_perf_all_SCAFFOLD_2, hyperparameters_SCAFFOLD_2] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                         batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                         print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                         init_model=init_model, save_period=save_period,
                                         lr_decay_per_round=lr_decay_per_round, 
                                         hpo_method=hpo_method, hpo_n_minibatch=hpo_n_minibatch,
                                         hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)
    
####

hpo_method = orig_hpo_method

print('FedAvg_%s'%hpo_method)

[fed_mdls_sel_FedAvg, trn_perf_sel_FedAvg, tst_perf_sel_FedAvg,
 fed_mdls_all_FedAvg, trn_perf_all_FedAvg,
 tst_perf_all_FedAvg, hyperparameters_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, 
                                     learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round, 
                                     hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)

hpo_method = hpo_method_2

print('FedAvg_%s'%hpo_method)

[fed_mdls_sel_FedAvg_2, trn_perf_sel_FedAvg_2, tst_perf_sel_FedAvg_2,
 fed_mdls_all_FedAvg_2, trn_perf_all_FedAvg_2,
 tst_perf_all_FedAvg_2, hyperparameters_FedAvg_2] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, 
                                     learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round, 
                                     hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)
        
#### 
# print('FedProx_%s'%hpo_method)

# mu = 1e-4

# [fed_mdls_sel_FedProx, trn_perf_sel_FedProx, tst_perf_sel_FedProx,
#  fed_mdls_all_FedProx, trn_perf_all_FedProx,
#  tst_perf_all_FedProx, hyperparameters_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, 
#                                      learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, save_period=save_period,
#                                      mu=mu, lr_decay_per_round=lr_decay_per_round, 
#                                      hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)


# Plot results
plt.figure(figsize=(6, 5))
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,0], label='FedDyn_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn_2[:,0], label='FedDyn_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, tst_perf_all_SCAFFOLD[:,0], label='SCAFFOLD_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, tst_perf_all_SCAFFOLD_2[:,0], label='SCAFFOLD_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg[:,0], label='FedAvg_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg_2[:,0], label='FedAvg_%s'%hpo_method_2)

# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx[:,0], label='FedProx_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx_2[:,0], label='FedProx_%s'%hpo_method_2)

plt.ylabel('Loss', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=12)#, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.title("Learning rate for Synthetic IID From different starting learning rates")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/lossplot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show()


# Plot results
plt.figure(figsize=(6, 5))
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,0], label='FedDyn_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn_2[:,0], label='FedDyn_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, tst_perf_all_SCAFFOLD[:,1], label='SCAFFOLD_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, tst_perf_all_SCAFFOLD_2[:,1], label='SCAFFOLD_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg[:,1], label='FedAvg_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg_2[:,1], label='FedAvg_%s'%hpo_method_2)

# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx[:,0], label='FedProx_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx_2[:,0], label='FedProx_%s'%hpo_method_2)

plt.ylabel('Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=12)#, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.title("Accuracy with Synthetic IID From different starting learning rates")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/accplot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show()


plt.figure(figsize=(6, 5))
# plt.plot(np.arange(com_amount)+1, hyperparameters_FedFyn[:,0], label='FedDyn_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, hyperparameters_FedFyn_2[:,0], label='FedDyn_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, hyperparameters_SCAFFOLD[:,0], label='SCAFFOLD_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, hyperparameters_SCAFFOLD_2[:,0], label='SCAFFOLD_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, hyperparameters_FedAvg[:,0], label='FedAvg_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, hyperparameters_FedAvg_2[:,0], label='FedAvg_%s'%hpo_method_2)

# plt.plot(np.arange(com_amount)+1, hyperparameters_FedProx[:,0], label='FedProx_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, hyperparameters_FedProx[:,0], label='FedProx_%s'%hpo_method_2)

plt.ylabel('Learning rates', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
#plt.legend(fontsize=10)#, loc='upper right', bbox_to_anchor=(1.015, -0.02))
plt.legend()
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.title("Loss with Synthetic IID From different starting learning rates")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/hyperparamsplot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')


print('final accuracy scaf nm',tst_perf_all_SCAFFOLD[399:,1])
print('final accuracy scaf reg',tst_perf_all_SCAFFOLD_2[399:,1])
print('final accuracy fedavg nm',tst_perf_all_FedAvg[399:,1])
print('final accuracy fedavg reg',tst_perf_all_FedAvg_2[399:,1])