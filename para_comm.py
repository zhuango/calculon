import json

def cal_w_optm_and_act_mem(calculon_json, model_json, run_json, fp32_grad=True, zero_stage=1, include_act_grad=False):
    run = json.load(open(run_json, 'r'))
    pp_size = run['pipeline_par']
    tp_size = run['tensor_par']
    dp_size = run['data_par']

    model = json.load(open(model_json, 'r'))
    num_blocks  = model['num_blocks']
    hidden_size = model['hidden']
    vocab_size  = model['vocab_size']
    tie_emb     = model['tie_emb']
    seql        = model['seq_size']

    d = json.load(open(calculon_json, 'r'))
    # (25329664+25329664*2+151977984)/1024/1024 * 32 + 32000/16 * 4096 * (18)*2/1024/1024
    
    # optimizer
    w_optim = d['optimizer_space']/(dp_size if zero_stage>=1 else 1)
    
    # weights gradient
    gradient = d['block_weight_grad_space'] * (2 if fp32_grad else 1) /(dp_size if zero_stage>=2 else 1)    
    w_optim += gradient * (num_blocks)/pp_size
    # w_optim += d['weight_grad_space']*(2 if fp32_grad else 1)
    
    # weights
    w_optim += d['weight_space']/(dp_size if zero_stage>=3 else 1)

    # embedding and lm_head space
    w_optim += hidden_size * vocab_size * (1 if tie_emb else 2) * (18 if fp32_grad else 16) / tp_size

    act = d['act_space']
    if include_act_grad:
        act += d['act_grad_space']
    # lm_head input and ce loss
    act += 4*(vocab_size*seql + hidden_size*seql) / tp_size


    # print(w_optim, act)
    return w_optim, act, d

def test():
    w_optim, act = cal_w_optm_and_act_mem('calculon_stats.json', './models/llama2-7b.json', 'examples/3072_t1_p1_d1_mbs1_full.json')
    M = 1024**2
    G = M*1024
    print(w_optim/M, "MB for w, w_grad and optim")
    print(act/M, "MB for act")
    print((w_optim + act)/M, "MB in total")


if __name__ == "__main__":
    # [data_par = [1,2,4,8]],
    # [pipeline_par = [1,2,4,8,16]],

    configs = [
        [1, 1, 16, 2048, 1],
        [1, 1, 16, 2048, 2],
        [1, 1, 16, 2048, 4],
        [1, 1, 16, 4096, 1],
        [1, 1, 16, 4096, 2],
        [1, 1, 16, 4096, 4],
        [2, 1, 16, 2048, 1],
        [2, 1, 16, 2048, 2],
        [2, 1, 16, 2048, 4],
        [2, 1, 16, 4096, 1],
        [2, 1, 16, 4096, 2],
        [2, 1, 16, 4096, 4],
        [2, 2, 16, 4096, 1],
        [2, 2, 16, 4096, 2],
        [2, 2, 16, 4096, 4],
        [4, 2, 16, 4096, 1],
        [4, 2, 16, 4096, 2],
        [4, 2, 16, 4096, 4],
        [4, 4, 16, 4096, 1],
        [4, 4, 16, 4096, 2],
        [4, 4, 16, 4096, 4],
        [8, 8, 16, 4096, 1],
        [8, 8, 16, 4096, 2],
        [8, 8, 16, 4096, 4],
        [8, 16, 8, 4096, 1],
        [8, 16, 8, 4096, 2],
        [8, 16, 8, 4096, 4],
    ]
    M = 1024**2
    G = M*1024
    
    master_exe_json = './examples/3072_t1_p1_d1_mbs1_full.json'

    master_app_json = './models/llama3-8b.json'
    master_app_json = './models/llama2-7b.json'
    all_details = []
    print("dp:pp:tp:seql:mbs:weight and optimizer MF(MB):activation MF(MB):total MF(MB):block_fw_tp_size(MB):block_bw_tp_size(MB):block_fw_pp_size(MB):block_bw_pp_size(MB)")
    for config in configs:
        dp, pp, tp, seql, mbs = config
        master_exe_config = json.load(open(master_exe_json, 'r'))
        master_exe_config['data_par'] = dp
        master_exe_config['pipeline_par'] = pp
        master_exe_config['tensor_par'] = tp
        master_exe_config['microbatch_size'] = mbs
        master_exe_config['batch_size'] = mbs * dp
        master_exe_config['num_procs'] = dp*pp*tp

        exe_json_name = "seql{}_dp{}_pp{}_tp{}_mbs{}.json".format(seql, dp, pp, tp, mbs)
        json.dump(master_exe_config, open(exe_json_name, 'w'))

        master_app_config = json.load(open(master_app_json, 'r'))
        master_app_config['seq_size'] = seql

        
        app_json_name = "model_seql{}.json".format(seql)
        json.dump(master_app_config, open(app_json_name, 'w'))
        import os
        os.system('python3 ./bin/calculon \
            llm {} \
            {} \
            ./systems/a100_80e.json \
            ./calculon_stats.json \
            -p ./calculon_peers.json \
            --layers'.format(app_json_name, exe_json_name))
        w_optim, act, details = cal_w_optm_and_act_mem('calculon_stats.json', app_json_name, exe_json_name)
        # print("=======================================================")
        print(
            dp, 
            pp, 
            tp, 
            seql, 
            mbs, 
            w_optim/M, 
            act/M, 
            (w_optim + act)/M, 
            details['baseblock_fw_tp_size']/M,
            details['baseblock_bw_tp_size']/M,
            details['block_fw_pp_size']/M,
            details['block_bw_pp_size']/M)
        all_details.append(details)
    print_name = True
    for detail in all_details:
        layer_names = []
        ad = []
        for layer in detail['layers']:
            if layer['fw_arithmetic_intensity'] != 0:
                layer_names.append(layer['name'])
                ad.append(layer['fw_arithmetic_intensity'])
        if print_name:
            print(":".join([str(elem) for elem in layer_names]))
            print_name = False
        print(":".join([str(elem) for elem in ad]))
