res_dirs = {
    '~/workspace/cafa4/evaluation/';
    '~/workspace/cafa3/evaluation/';
    '~/workspace/cafa2/evaluation/';
    '~/workspace/cafa1/evaluation/';
};

cafas = {
    'cafa4';
    'cafa3';
    'cafa2';
    'cafa1';
};

cfg_dirs = {
    '~/workspace/cafa4/config/';
    '~/workspace/cafa3/config_cafa4bm/';
    '~/workspace/cafa2/config_cafa4bm/';
    '~/workspace/cafa1/config_cafa4bm/';
};

ontnames = { 'mfo', 'bpo', 'cco' };

% output_dir = '~/workspace/cafa4/cafas_comparison_evaluation_42/';
output_dir = '~/workspace/cafa4/cafas_comparison_evaluation_41/';

mkdir(output_dir)

ntop = 10

colors = {
    [0.6, 0.4, 0.6];
    [0.6, 0.6, 0.4];
    [0.4, 0.6, 0.6];
    [0.5, 0.6, 0.5];
}

bcolor = {[196,  48,  43]/255, [  0,  83, 159]/255}; % baseline models

btxtcolor = {[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]}; % for baseline models


yranges = {
    [0.1, 0.8];
    [0, 0.5];
    [0.2, 0.8];
};

yranges = {
    [0.0, 0.8];
    [0.0, 0.8];
    [0.0, 0.8];
};

for o = 1:numel(ontnames)
    figure
    ontname = ontnames{o}
    model_scores = {};
    groups = {};
    groups_top = {};
    for i = 1:numel(res_dirs)
        res_dir = res_dirs{i}
        cfg_dir = cfg_dirs{i};
        model_scores{i} = [];
        
        ontres_dir = [res_dir, ontname, '_all_type1_mode1/']
        
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
        

        groups{i} = [model_scores{:,i}.score];
        groups{i} = {groups{i}.seq_fmax_bst};
        group_array = [groups{i}{:}];
        [group_sel, ~, info] = cafa_sel_top_seq_fmax(ntop, groups{i}, '', '', [cfg_dirs{1}, 'register.tab'], false);
        [~, idx] = ismember(info.top_mid, {group_array.id});
        group_top = group_array(idx);
        
        for ii = 1:ntop
            if ii > numel(group_sel)
                break
            end
            groups_top(i, ii).group_sel = group_sel{ii};
            groups_top(i, ii).top_mid = info.top_mid{ii};
            groups_top(i, ii).idx = idx(ii);
            groups_top(i, ii).avgfmaxs  = nanmean([group_top(ii).fmax_bst]);
            % matrix percentile
            groups_top(i, ii).fmax_q05  = prctile([group_top(ii).fmax_bst], 5);
            groups_top(i, ii).fmax_q95  = prctile([group_top(ii).fmax_bst], 95);
            groups_top(i, ii).coverage  = nanmean([group_top(ii).coverage_bst]);
%             name_rank = arrayfun(@num2str, [1:ntop]', 'UniformOutput', 0);
            groups_top(i, ii).disp_name = strcat(cafas{i}, '_', num2str(ii));
            groups_top(i, ii).color = colors{i};
        end
    end
    
    all_in_one = reshape(groups_top', [10*size(groups_top, 1), 1]);
%     all_in_one = [];
%     fields = fieldnames(groups_top);
%     for k = 1:numel(fields)
%         aField     = fields{k}; % EDIT: changed to {}
%         all_in_one.(aField) = cat(1, groups_top.(aField));
%     end

    [~, rank_idx] = sort([all_in_one.avgfmaxs], 'descend');
    data = all_in_one(rank_idx);
    bar_w  = 0.7;
    base_fs = 10; % base font size
    ylim = yranges{o};
    ylim_l = ylim(1);
    ylim_u = ylim(2);
    for i = 1:ntop
        rpos = [i - bar_w / 2, ylim_l, bar_w, data(i).avgfmaxs - ylim_l];
        rectangle('Position', rpos, 'FaceColor', data(i).color, 'EdgeColor', data(i).color);
        line([i, i], [data(i).fmax_q05, data(i).fmax_q95], 'Color', 'k');
        line([i - bar_w / 4, i + bar_w / 4], [data(i).fmax_q05, data(i).fmax_q05], 'Color', 'k');
        line([i - bar_w / 4, i + bar_w / 4], [data(i).fmax_q95, data(i).fmax_q95], 'Color', 'k');

        % plot coverage as text
        cpos  = ylim_l + 0.05 * (ylim_u - ylim_l);
        ctext = sprintf('C=%.2f', nanmean(data(i).coverage));
        text(i, cpos, ctext, 'FontSize', base_fs, 'Rotation', 90);

        % collect team name for display
        xticks{i} = regexprep(data(i).disp_name, '_', '\\_');
    end


%     bsl_res{1} = bsl_scores{1,1}.score.seq_fmax_bst;
%     bsl_res{2} = bsl_scores{2,1}.score.seq_fmax_bst;
%     bsl_res{3} = bsl_scores{1,2}.score.seq_fmax_bst;
%     bsl_res{4} = bsl_scores{2,2}.score.seq_fmax_bst;
%     bsl_res{5} = bsl_scores{1,3}.score.seq_fmax_bst;
%     bsl_res{6} = bsl_scores{2,3}.score.seq_fmax_bst;
%     bsl_res{7} = bsl_scores{1,4}.score.seq_fmax_bst;
%     bsl_res{8} = bsl_scores{2,4}.score.seq_fmax_bst;
    bsl_scores = bsl_scores';
    for i = 1:numel(res_dirs)
        b = 1;
        bsl_res = bsl_scores{b, i}.score.seq_fmax_bst;
        bsl_data.avgfmaxs = nanmean(bsl_res.fmax_bst);
        bsl_data.fmax_q05  = prctile([bsl_res.fmax_bst], 5);
        bsl_data.fmax_q95  = prctile([bsl_res.fmax_bst], 95);
        bsl_data.coverage  = nanmean([bsl_res.coverage_bst]);
%             name_rank = arrayfun(@num2str, [1:ntop]', 'UniformOutput', 0);
        bsl_data.disp_name = strcat(cafas{i}, '_Naive');
        bsl_data.color = bcolor{b};
        j = ntop + (i - 1)*2 + b;
        rpos = [j - bar_w / 2, ylim_l, bar_w, bsl_data.avgfmaxs - ylim_l];
        rectangle('Position', rpos, 'FaceColor', bsl_data.color, 'EdgeColor', bsl_data.color);
        line([j, j], [bsl_data.fmax_q05, bsl_data.fmax_q95], 'Color', 'k');
        line([j - bar_w / 4, j + bar_w / 4], [bsl_data.fmax_q05, bsl_data.fmax_q05], 'Color', 'k');
        line([j - bar_w / 4, j + bar_w / 4], [bsl_data.fmax_q95, bsl_data.fmax_q95], 'Color', 'k');

        % plot coverage as text
        cpos  = ylim_l + 0.05 * (ylim_u - ylim_l);
        ctext = sprintf('C=%.2f', nanmean(bsl_data.coverage));
        text(j, cpos, ctext, 'Rotation', 90, 'FontSize', base_fs, 'Color', btxtcolor{b});

        % collect team name for display
        xticks{j} = regexprep(bsl_data.disp_name, '_', '\\_');
        
        b = 2;
        bsl_res = bsl_scores{b, i}.score.seq_fmax_bst;
        bsl_data.avgfmaxs = nanmean(bsl_res.fmax_bst);
        bsl_data.fmax_q05  = prctile([bsl_res.fmax_bst], 5);
        bsl_data.fmax_q95  = prctile([bsl_res.fmax_bst], 95);
        bsl_data.coverage  = nanmean([bsl_res.coverage_bst]);
%             name_rank = arrayfun(@num2str, [1:ntop]', 'UniformOutput', 0);
        bsl_data.disp_name = strcat(cafas{i}, '_Blast');
        bsl_data.color = bcolor{b};
        
        j = ntop + (i - 1)*2 + b;
        rpos = [j - bar_w / 2, ylim_l, bar_w, bsl_data.avgfmaxs - ylim_l];
        rectangle('Position', rpos, 'FaceColor', bsl_data.color, 'EdgeColor', bsl_data.color);
        line([j, j], [bsl_data.fmax_q05, bsl_data.fmax_q95], 'Color', 'k');
        line([j - bar_w / 4, j + bar_w / 4], [bsl_data.fmax_q05, bsl_data.fmax_q05], 'Color', 'k');
        line([j - bar_w / 4, j + bar_w / 4], [bsl_data.fmax_q95, bsl_data.fmax_q95], 'Color', 'k');

        % plot coverage as text
        cpos  = ylim_l + 0.05 * (ylim_u - ylim_l);
        ctext = sprintf('C=%.2f', nanmean(bsl_data.coverage));
        text(j, cpos, ctext, 'Rotation', 90, 'FontSize', base_fs, 'Color', btxtcolor{b});

        % collect team name for display
        xticks{j} = regexprep(bsl_data.disp_name, '_', '\\_');

    end
    
    title(upper(ontname));
    ylabel('Fmax');
    
    ax = gca;
    ax.XLim               = [0, (ntop + numel(res_dirs)*2 + 1)];
    ax.YLim               = [ylim_l, ylim_u];
    ax.XTick              = 1 : (ntop + numel(res_dirs)*2);
    ax.YTick              = ylim_l : 0.1 : ylim_u;
    ax.XTickLabel         = xticks;
    ax.FontSize           = base_fs;
    ax.XTickLabelRotation = 45;
    
    
%     cafa_plot_duel_fmax([output_dir, ontname, '_duel_result.png'],duel_res,bsl_res,ontname)
    saveas(gcf,[output_dir, ontname, '_cafas.png'])
    close
end

