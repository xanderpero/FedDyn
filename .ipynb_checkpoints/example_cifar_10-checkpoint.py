from utils_general import *
from utils_methods import *

# Dataset initialization

########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
########
# For 'CIFAR100' experiments
#     - Change the dataset argument from emnist to CIFAR100.
########
# For 'CIFAR10' experiments
#     - Change the dataset argument from emnist to CIFAR10.
########
# For 'mnist' experiments
#     - Change the dataset argument from emnist to mnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########





n_client = 100

##### Generate IID
# data_obj = DatasetObject(dataset='emnist', n_client=n_client, rule='iid', unbalanced_sgm=0)

##### Dirichlet(0.3)
# data_obj = DatasetObject(dataset='emnist', n_client=n_client, rule='Dirichlet', rule_arg=0.3, unbalanced_sgm=0)

##### Dirichlet(0.6)
data_obj = DatasetObject(dataset='emnist', n_client=n_client, rule='Dirichlet', rule_arg=0.6, unbalanced_sgm=0)

###
model_name         = 'emnist' # Model type
com_amount         = 200
save_period        = 100
weight_decay       = 1e-4
batch_size         = 50
act_prob           = 0.1  # % participating clients
# lr_decay_per_round = 1    # do not change 
epoch              = 10   # local epochs
learning_rate      = 0.1
print_per          = 20
hpo_method         = 'nm' # 'nm' 'hpona' 'decay'
hpo_method_2       = 'decay' # 'hpona' 'decay'
hpo_epoch          = 10    # number of steps with hyperparameter configuration
hpo_maxiter        = 5    # number of comparisons with Nelder-Mead
hpo_per            = 40   # perform HPO every X communication rounds

if hpo_method == 'decay':
    lr_decay_per_round = 0.998
else:
    lr_decay_per_round = 1

orig_hpo_method = hpo_method
hpo_method_2 = hpo_method_2

# Model function
model_func = lambda : client_model(model_name)
init_model = model_func()

# Initalise the model for all methods or load it from a saved initial model
if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    print(data_obj.name)
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
    
    
print(data_obj.name)
    
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

# hpo_method = 'decay'
# if hpo_method == 'decay':
#     lr_decay_per_round = 0.998
# print('FedDyn_%s'%hpo_method)

# alpha_coef = 0.001
# [fed_mdls_sel_FedFyn_2, trn_perf_sel_FedFyn_2, tst_perf_sel_FedFyn_2,
#  fed_mdls_all_FedFyn_2, trn_perf_all_FedFyn_2, tst_perf_all_FedFyn_2,
#  fed_mdls_cld_FedFyn_2, hyperparameters_FedFyn_2] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
#                                      learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
#                                      save_period=save_period, lr_decay_per_round=lr_decay_per_round, 
#                                      hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)

###

hpo_method = orig_hpo_method
lr_decay_per_round = 1
    
print('SCAFFOLD_%s'%hpo_method)
n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
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
if hpo_method == 'decay':
    lr_decay_per_round = 0.998

print('SCAFFOLD_%s'%hpo_method)
    
[fed_mdls_sel_SCAFFOLD_2, trn_perf_sel_SCAFFOLD_2, tst_perf_sel_SCAFFOLD_2,
 fed_mdls_all_SCAFFOLD_2, trn_perf_all_SCAFFOLD_2,
 tst_perf_all_SCAFFOLD_2, hyperparameters_SCAFFOLD_2] = train_SCAFFOLD(data_obj=data_obj, 
                                         act_prob=act_prob, learning_rate=learning_rate,
                                         batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                         print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                         init_model=init_model, save_period=save_period,
                                         lr_decay_per_round=lr_decay_per_round, 
                                         hpo_method=hpo_method, hpo_n_minibatch=hpo_n_minibatch, 
                                         hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)

    
####
hpo_method = orig_hpo_method
lr_decay_per_round = 1
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

# hpo_method = orig_hpo_method
# print('FedProx_%s'%hpo_method)
# lr_decay_per_round = 1
# mu = 1e-4

# [fed_mdls_sel_FedProx, trn_perf_sel_FedProx, tst_perf_sel_FedProx,
#  fed_mdls_all_FedProx, trn_perf_all_FedProx,
#  tst_perf_all_FedProx, hyperparameters_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, 
#                                      learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, save_period=save_period,
#                                      mu=mu, lr_decay_per_round=lr_decay_per_round, 
#                                      hpo_method=hpo_method, hpo_epoch=hpo_epoch, hpo_maxiter=hpo_maxiter, hpo_per=hpo_per)


# hpo_method = hpo_method_2
# if hpo_method == 'decay':
#     lr_decay_per_round = 0.998
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


#####################################################

# Plot results
plt.figure(figsize=(6, 5))
#plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,0], label='FedDyn_%s'%orig_hpo_method)
#plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn_2[:,0], label='FedDyn_%s'%hpo_method_2)

plt.plot(np.arange(com_amount-25)+1+25, tst_perf_all_SCAFFOLD[25:,0], label='SCAFFOLD_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount-25)+1+25, tst_perf_all_SCAFFOLD_2[25:,0], label='SCAFFOLD_%s'%hpo_method_2)

plt.plot(np.arange(com_amount-25)+1+25, tst_perf_all_FedAvg[25:,0], label='FedAvg_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount-25)+1+25, tst_perf_all_FedAvg_2[25:,0], label='FedAvg_%s'%hpo_method_2)

# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx[:,0], label='FedProx_%s'%sorig_hpo_method)
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx_2[:,0], label='FedProx_%s'%hpo_method_2)


plt.ylabel('Loss', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16) #, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.xlim([com_amount+1-175, com_amount+1])


plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/lossplot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')

# Plot results
plt.figure(figsize=(6, 5))
#plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,1], label='FedDyn_%s'%orig_hpo_method)
#plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn_2[:,1], label='FedDyn_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, tst_perf_all_SCAFFOLD[:,1], label='SCAFFOLD_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, tst_perf_all_SCAFFOLD_2[:,1], label='SCAFFOLD_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg[:,1], label='FedAvg_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg_2[:,1], label='FedAvg_%s'%hpo_method_2)

# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx[:,1], label='FedProx_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedProx_2[:,1], label='FedProx_%s'%hpo_method_2)


plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/accplot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show() 


plt.figure(figsize=(6, 5))
#plt.plot(np.arange(com_amount)+1, hyperparameters_FedFyn[:,0], label='FedDyn_%s'%orig_hpo_method)
#plt.plot(np.arange(com_amount)+1, hyperparameters_FedFyn_2[:,0], label='FedDyn_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, hyperparameters_SCAFFOLD[:,0], label='SCAFFOLD_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, hyperparameters_SCAFFOLD_2[:,0], label='SCAFFOLD_%s'%hpo_method_2)

plt.plot(np.arange(com_amount)+1, hyperparameters_FedAvg[:,0], label='FedAvg_%s'%orig_hpo_method)
plt.plot(np.arange(com_amount)+1, hyperparameters_FedAvg_2[:,0], label='FedAvg_%s'%hpo_method_2)

# plt.plot(np.arange(com_amount)+1, hyperparameters_FedProx[:,0], label='FedProx_%s'%orig_hpo_method)
# plt.plot(np.arange(com_amount)+1, hyperparameters_FedProx_2[:,0], label='FedProx_%s'hpo_method_2)


plt.ylabel('Learning Rate', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/hyperplot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
