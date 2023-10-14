res_dirs = {
%     '~/workspace/cafa1/evaluation/';
%     '~/workspace/cafa2/evaluation/';
    '~/workspace/cafa3/evaluation/';
    '~/workspace/cafa4/evaluation/';
}

cafas = {
    'cafa3', 'cafa4'
}

cfg_dirs = {
%     '~/workspace/cafa1/config_cafa4/';
%     '~/workspace/cafa2/config_cafa4bm/';
    '~/workspace/cafa3/config_cafa4bm/';
    '~/workspace/cafa4/config/';
}

ontnames = { 'mfo', 'bpo', 'cco' }

output_dir = '~/workspace/cafa4/cafas_comparison_evaluation/'

bm_type = 'type1'

metric = 'smin'

mkdir(output_dir)

for j = 1:numel(ontnames)
    figure
    ontname = ontnames{j}
    model_scores = {};
    for i = 1:numel(res_dirs)
        res_dir = res_dirs{i}
        cfg_dir = cfg_dirs{i};
        model_scores{i} = [];
        
        ontres_dir = [res_dir, ontname, '_all_', bm_type, '_mode1/']
        
        models = dir([ontres_dir, 'M*']);
        for k = 1:numel(models)
            model = models(k).name;
            model = split(model, '.');
            model = model{1};
            disp(model);
%             model = model.name
            model_score.name = [cafas{i}, '_', model];
            model_score.score = load([ontres_dir, model, '.mat']);
%             model_score.color = colors{i};
            
            model_scores{i} = [model_scores{i}; model_score];
        end
        
        bsl_scores{i,1}.name = [cafas{i}, '_BN4S'];
        bsl_scores{i,1}.score = load([ontres_dir, 'BN4S.mat']);
        
        bsl_scores{i,2}.name = [cafas{i}, '_BB4S'];
        bsl_scores{i,2}.score = load([ontres_dir, 'BB4S.mat']);
    end

    group1 = [model_scores{:,1}.score]
    group1 = {group1.seq_smin_bst}
    group1_array = [group1{:}]
    [group1_sel, ~, info] = cafa_sel_top_seq_smin(5, group1, '', '', [cfg_dirs{1}, 'register.tab'], false);
    [~, idx] = ismember(info.top_mid, {group1_array.id})
%     avgsmins = nanmean([group1.smin_bst])
%     [~,idx] = sort(avgsmins, 'descend');
    for i = 1:5
        group1_top5{i} = group1{idx(i)};
    end

    group2 = [model_scores{:,2}.score]
    group2 = {group2.seq_smin_bst}
    group2_array = [group2{:}]
    [group2_sel, ~, info] = cafa_sel_top_seq_smin(5, group2, '', '', [cfg_dirs{2}, 'register.tab'], false);
    [~, idx] = ismember(info.top_mid, {group2_array.id})
%     avgsmins = nanmean([group2.smin_bst])
%     [~,idx] = sort(avgsmins, 'descend');
    for i = 1:5
        group2_top5{i} = group2{idx(i)};
    end

    duel_res = cafa_duel_seq_smin(group1_top5, group2_top5)
    
    bsl_res = {};
    bsl_res{1} = bsl_scores{1,1}.score.seq_smin_bst;
    bsl_res{2} = bsl_scores{2,1}.score.seq_smin_bst;
    bsl_res{3} = bsl_scores{1,2}.score.seq_smin_bst;
    bsl_res{4} = bsl_scores{2,2}.score.seq_smin_bst;
    
    cafa_plot_duel_smin([output_dir, ontname, '_duel_result_smin.png'],duel_res,bsl_res,ontname)
    close
end
