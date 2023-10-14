
bm = readcell('~/workspace/cafa4/annotations/benchmark/lists/mfo_all_type1.txt');

% config = cafa_parse_config('~/workspace/cafa4/config/preeval/mfo_preeval.job');
config = cafa_parse_config('~/workspace/cafa3/config/preeval/mfo_preeval.job');
load('~/workspace/cafa4/annotations/benchmark/groundtruth/mfoa.mat');

% pred_dir = '~/workspace/cafa/prediction/';
pred_dir = '~/workspace/cafa3/prediction/';


all_fmaxs = [];
all_taus = [];
for i = 1:numel(config.model)
    mid = config.model{i};
    prrec.model = mid;

    load([pred_dir, 'mfo/',mid,'.mat']);
    [~, ev_index] = ismember(bm, pred.object);
    haspred_flags = ev_index~=0;
    [~, bm_index] = ismember(bm, oa.object);
    ref = oa.annotation(bm_index(haspred_flags),:);
    eval_pred = pred.score(ev_index(haspred_flags),:);
    
%     load(['~/workspace/cafa/evaluation/mfo_all_type1_mode1/',mid,'.mat']);
%     tau = seq_fmax.tau;
    
    tau_range = (0:0.01:1)';
%     [precisions, recalls] = pr_curves(eval_pred, ref, tau_range);
%     [precisions, recalls]
    curves = pr_curves(eval_pred, ref, tau_range);
    target_fmaxs = [];
    taus = [];
    for ti = 1:size(ref,1)
        [target_fmaxs(ti), ~, taus(ti)] = pfp_fmaxc(curves{ti}, tau_range);
    end
    target_fmaxs
    all_fmaxs(haspred_flags,i) = target_fmaxs';
    all_taus(haspred_flags,i) = taus';
    
    histogram(target_fmaxs, 20);
    saveas(gcf, ['~/workspace/cafa3/evaluation/mfo_all_type1_mode1/models/', mid, '.png']);
end

% nonan_all_fmaxs = all_fmaxs;
% nonan_all_fmaxs(isnan(all_fmaxs)) = 0;
% mean_fmaxs = mean(nonan_all_fmaxs, 2);
mean_fmaxs = mean(all_fmaxs, 2);
[sorted_mean_fmaxs, sind] = sort(mean_fmaxs);
sorted_all_fmaxs = all_fmaxs(sind,:);
boxplot(sorted_all_fmaxs')

leaf_oa = pfp_leafannot(oa);

figure;

nbm = numel(bm);

easiest.bm_ind = sind(nbm-3);
easiest.target = bm{easiest.bm_ind}

hardest.bm_ind = sind(3);
hardest.target = bm{hardest.bm_ind}

medium.bm_ind = sind(floor(nbm/2));
medium.target = bm{medium.bm_ind}

%%
[~, easiest.target_ind] = ismember(easiest.target, oa.object)
easiest.truth.terms = oa.ontology.term(oa.annotation(easiest.target_ind, :))

easiest.fmaxs = all_fmaxs(easiest.bm_ind, :);
easiest.fmaxs(easiest.fmaxs==0) = NaN;

worst = {};
[worst.fmax, worst.model_ind] = min(easiest.fmaxs)
worst.model = config.model{worst.model_ind}
worst.pred = load(['~/workspace/cafa/prediction/mfo/',worst.model,'.mat']);
worst.pred = worst.pred.pred;
[~, ind] = ismember(easiest.target, worst.pred.object)
worst.pred = worst.pred.score(ind, :);
worst.tau = all_taus(easiest.bm_ind, worst.model_ind)
worst.terms = oa.ontology.term(worst.pred >= worst.tau)
easiest.worst = worst

naive = {};
[~, naive.model_ind] = ismember('BN4S', config.model)
naive.fmax = easiest.fmaxs(naive.model_ind)
naive.model = config.model{naive.model_ind}
naive.pred = load(['~/workspace/cafa/prediction/mfo/',naive.model,'.mat']);
naive.pred = naive.pred.pred;
[~, ind] = ismember(easiest.target, naive.pred.object)
naive.pred = naive.pred.score(ind, :);
naive.tau = all_taus(easiest.bm_ind, naive.model_ind)
naive.terms = oa.ontology.term(naive.pred >= naive.tau)
easiest.naive = naive;

blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
blast.fmax = easiest.fmaxs(blast.model_ind)
blast.model = config.model{blast.model_ind}
blast.pred = load(['~/workspace/cafa/prediction/mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(easiest.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(easiest.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
easiest.blast = blast;

M115 = {};
[~, M115.model_ind] = ismember('M115', config.model)
M115.fmax = easiest.fmaxs(M115.model_ind)
M115.model = config.model{M115.model_ind}
M115.pred = load(['~/workspace/cafa/prediction/mfo/',M115.model,'.mat']);
M115.pred = M115.pred.pred;
[~, ind] = ismember(easiest.target, M115.pred.object)
M115.pred = M115.pred.score(ind, :);
M115.tau = all_taus(easiest.bm_ind, M115.model_ind)
M115.terms = oa.ontology.term(M115.pred >= M115.tau)
easiest.M115 = M115

M037 = {};
[~, M037.model_ind] = ismember('M037', config.model)
M037.fmax = easiest.fmaxs(M037.model_ind)
M037.model = config.model{M037.model_ind}
M037.pred = load(['~/workspace/cafa/prediction/mfo/',M037.model,'.mat']);
M037.pred = M037.pred.pred;
[~, ind] = ismember(easiest.target, M037.pred.object);
M037.pred = M037.pred.score(ind, :);
M037.tau = all_taus(easiest.bm_ind, M037.model_ind)
M037.terms = oa.ontology.term(M037.pred >= M037.tau)
easiest.M037 = M037

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = easiest.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load(['~/workspace/cafa/prediction/mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(easiest.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(easiest.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
easiest.M082 = M082

M006 = {};
[~, M006.model_ind] = ismember('M006', config.model)
M006.fmax = easiest.fmaxs(M006.model_ind)
M006.model = config.model{M006.model_ind}
M006.pred = load(['~/workspace/cafa/prediction/mfo/',M006.model,'.mat']);
M006.pred = M006.pred.pred;
[~, ind] = ismember(easiest.target, M006.pred.object);
M006.pred = M006.pred.score(ind, :);
M006.tau = all_taus(easiest.bm_ind, M006.model_ind)
M006.terms = oa.ontology.term(M006.pred >= M006.tau)
easiest.M006 = M006

%%
blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
for i = 1:nbm
    best = {}
    easy_exblast.bm_ind = sind(nbm-i+1);
    blast.fmax = all_fmaxs(easy_exblast.bm_ind, blast.model_ind)
    easy_exblast.target = bm{easy_exblast.bm_ind}
    best.fmax = max(all_fmaxs(easy_exblast.bm_ind, :))
    if best.fmax > blast.fmax
        break
    end
end

easy_exblast.target = 'T96060007155';

[~, easy_exblast.target_ind] = ismember(easy_exblast.target, oa.object)
easy_exblast.truth.terms = oa.ontology.term(oa.annotation(easy_exblast.target_ind, :))
easy_exblast.truth.leaf_terms = oa.ontology.term(leaf_oa(easy_exblast.target_ind, :))

easy_exblast.fmaxs = all_fmaxs(easy_exblast.bm_ind, :);
easy_exblast.fmaxs(easy_exblast.fmaxs==0) = NaN;

% worst = {};
% [worst.fmax, worst.model_ind] = min(easy_exblast.fmaxs)
% worst.model = config.model{worst.model_ind}
% worst.pred = load(['~/workspace/cafa/prediction/mfo/',worst.model,'.mat']);
% worst.pred = worst.pred.pred;
% [~, ind] = ismember(easy_exblast.target, worst.pred.object)
% worst.pred = worst.pred.score(ind, :);
% worst.tau = all_taus(easy_exblast.bm_ind, worst.model_ind)
% worst.terms = oa.ontology.term(worst.pred >= worst.tau)
% easy_exblast.worst = worst

best = {};
[best.fmax, best.model_ind] = max(easy_exblast.fmaxs)
best.model = config.model{best.model_ind}
best.pred = load(['~/workspace/cafa/prediction/mfo/',best.model,'.mat']);
best.pred = best.pred.pred;
[~, ind] = ismember(easy_exblast.target, best.pred.object)
best.pred = best.pred.score(ind, :);
best.tau = all_taus(easy_exblast.bm_ind, best.model_ind)
best.terms = oa.ontology.term(best.pred >= best.tau)
easy_exblast.best = best

naive = {};
[~, naive.model_ind] = ismember('BN4S', config.model)
naive.fmax = easy_exblast.fmaxs(naive.model_ind)
naive.model = config.model{naive.model_ind}
naive.pred = load(['~/workspace/cafa/prediction/mfo/',naive.model,'.mat']);
naive.pred = naive.pred.pred;
[~, ind] = ismember(easy_exblast.target, naive.pred.object)
naive.pred = naive.pred.score(ind, :);
naive.tau = all_taus(easy_exblast.bm_ind, naive.model_ind)
naive.terms = oa.ontology.term(naive.pred >= naive.tau)
naive.leaf_terms = oa.ontology.term(leaf_oa(easy_exblast.target_ind, naive.pred >= naive.tau))
easy_exblast.naive = naive;

blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
blast.fmax = easy_exblast.fmaxs(blast.model_ind)
blast.model = config.model{blast.model_ind}
blast.pred = load(['~/workspace/cafa/prediction/mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(easy_exblast.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(easy_exblast.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
easy_exblast.blast = blast;

M115 = {};
[~, M115.model_ind] = ismember('M115', config.model)
M115.fmax = easy_exblast.fmaxs(M115.model_ind)
M115.model = config.model{M115.model_ind}
M115.pred = load(['~/workspace/cafa/prediction/mfo/',M115.model,'.mat']);
M115.pred = M115.pred.pred;
[~, ind] = ismember(easy_exblast.target, M115.pred.object)
M115.pred = M115.pred.score(ind, :);
M115.tau = all_taus(easy_exblast.bm_ind, M115.model_ind)
M115.terms = oa.ontology.term(M115.pred >= M115.tau)
easy_exblast.M115 = M115

M037 = {};
[~, M037.model_ind] = ismember('M037', config.model)
M037.fmax = easy_exblast.fmaxs(M037.model_ind)
M037.model = config.model{M037.model_ind}
M037.pred = load(['~/workspace/cafa/prediction/mfo/',M037.model,'.mat']);
M037.pred = M037.pred.pred;
[~, ind] = ismember(easy_exblast.target, M037.pred.object);
M037.pred = M037.pred.score(ind, :);
M037.tau = all_taus(easy_exblast.bm_ind, M037.model_ind)
M037.terms = oa.ontology.term(M037.pred >= M037.tau)
easy_exblast.M037 = M037

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = easy_exblast.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load(['~/workspace/cafa/prediction/mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(easy_exblast.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(easy_exblast.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
easy_exblast.M082 = M082

M006 = {};
[~, M006.model_ind] = ismember('M006', config.model)
M006.fmax = easy_exblast.fmaxs(M006.model_ind)
M006.model = config.model{M006.model_ind}
M006.pred = load(['~/workspace/cafa/prediction/mfo/',M006.model,'.mat']);
M006.pred = M006.pred.pred;
[~, ind] = ismember(easy_exblast.target, M006.pred.object);
M006.pred = M006.pred.score(ind, :);
M006.tau = all_taus(easy_exblast.bm_ind, M006.model_ind)
M006.terms = oa.ontology.term(M006.pred >= M006.tau)
easy_exblast.M006 = M006

%%
[~, hardest.target_ind] = ismember(hardest.target, oa.object)
hardest.truth.terms = oa.ontology.term(oa.annotation(hardest.target_ind, :))

hardest.fmaxs = all_fmaxs(hardest.bm_ind, :);
hardest.taus = all_taus(hardest.bm_ind, :);

best = {};
[best.fmax, best.model_ind] = max(hardest.fmaxs)
best.model = config.model{best.model_ind}
best.pred = load(['~/workspace/cafa/prediction/mfo/',best.model,'.mat']);
best.pred = best.pred.pred;
[~, ind] = ismember(hardest.target, best.pred.object)
best.pred = best.pred.score(ind, :);
best.tau = all_taus(hardest.bm_ind, best.model_ind)
best.terms = oa.ontology.term(best.pred >= best.tau)
hardest.best = best

% naive = {};
% [~, naive.model_ind] = ismember('BN4S', config.model)
% naive.fmax = hardest.fmaxs(naive.model_ind)
% naive.model = config.model{naive.model_ind}
% naive.pred = load(['~/workspace/cafa/prediction/mfo/',naive.model,'.mat']);
% naive.pred = naive.pred.pred;
% [~, ind] = ismember(hardest.target, naive.pred.object)
% naive.pred = naive.pred.score(ind, :);
% naive.tau = all_taus(hardest.bm_ind, naive.model_ind)
% naive.terms = oa.ontology.term(naive.pred >= naive.tau)
% hardest.naive = naive;

blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
blast.fmax = hardest.fmaxs(blast.model_ind)
blast.model = config.model{blast.model_ind}
blast.pred = load(['~/workspace/cafa/prediction/mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(hardest.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(hardest.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
hardest.blast = blast
% hardest.blast = get_model_prediction_results('BB4S', hardest, config);

M115 = {};
[~, M115.model_ind] = ismember('M115', config.model)
M115.fmax = hardest.fmaxs(M115.model_ind)
M115.model = config.model{M115.model_ind}
M115.pred = load(['~/workspace/cafa/prediction/mfo/',M115.model,'.mat']);
M115.pred = M115.pred.pred;
[~, ind] = ismember(hardest.target, M115.pred.object)
M115.pred = M115.pred.score(ind, :);
M115.tau = all_taus(hardest.bm_ind, M115.model_ind)
M115.terms = oa.ontology.term(M115.pred >= M115.tau)
hardest.M115 = M115

M037 = {};
[~, M037.model_ind] = ismember('M037', config.model)
M037.fmax = hardest.fmaxs(M037.model_ind)
M037.model = config.model{M037.model_ind}
M037.pred = load(['~/workspace/cafa/prediction/mfo/',M037.model,'.mat']);
M037.pred = M037.pred.pred;
[~, ind] = ismember(hardest.target, M037.pred.object)
M037.pred = M037.pred.score(ind, :);
M037.tau = all_taus(hardest.bm_ind, M037.model_ind)
M037.terms = oa.ontology.term(M037.pred >= M037.tau)
hardest.M037 = M037

M006 = {};
[~, M006.model_ind] = ismember('M006', config.model)
M006.fmax = hardest.fmaxs(M006.model_ind)
M006.model = config.model{M006.model_ind}
M006.pred = load(['~/workspace/cafa/prediction/mfo/',M006.model,'.mat']);
M006.pred = M006.pred.pred;
[~, ind] = ismember(hardest.target, M006.pred.object)
M006.pred = M006.pred.score(ind, :);
M006.tau = all_taus(hardest.bm_ind, M006.model_ind)
M006.terms = oa.ontology.term(M006.pred >= M006.tau)
hardest.M006 = M006




