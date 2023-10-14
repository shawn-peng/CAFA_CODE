
bm = readcell('~/workspace/cafa4/annotations/benchmark/lists/mfo_all_type1.txt');

% config = cafa_parse_config('~/workspace/cafa4/config/preeval/mfo_preeval.job');
config = cafa_parse_config('~/workspace/cafa3/config/preeval/mfo_preeval.job');
% load('~/workspace/cafa4/annotations/benchmark/groundtruth/mfoa.mat');
load('~/workspace/cafa4/annotations/benchmark_intersection_cafa3/groundtruth/mfoa.mat');

% pred_dir = '~/workspace/cafa/prediction/';
pred_dir = '~/workspace/cafa3/prediction/';

mkdir('~/workspace/cafa3/evaluation/mfo_all_type1_mode1/models/')

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

medium1.bm_ind = sind(floor(nbm/2)+1);
medium1.target = bm{medium1.bm_ind}

medium2.bm_ind = sind(floor(nbm/2)-1);
medium2.target = bm{medium2.bm_ind}

%%
[~, easiest.target_ind] = ismember(easiest.target, oa.object)
easiest.truth.terms = oa.ontology.term(oa.annotation(easiest.target_ind, :))

easiest.fmaxs = all_fmaxs(easiest.bm_ind, :);
easiest.fmaxs(easiest.fmaxs==0) = NaN;

worst = {};
[worst.fmax, worst.model_ind] = min(easiest.fmaxs)
worst.model = config.model{worst.model_ind}
worst.pred = load([pred_dir, 'mfo/',worst.model,'.mat']);
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
naive.pred = load([pred_dir, 'mfo/',naive.model,'.mat']);
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
blast.pred = load([pred_dir, 'mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(easiest.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(easiest.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
easiest.blast = blast;

M064 = {};
[~, M064.model_ind] = ismember('M064', config.model)
M064.fmax = easiest.fmaxs(M064.model_ind)
M064.model = config.model{M064.model_ind}
M064.pred = load([pred_dir, 'mfo/',M064.model,'.mat']);
M064.pred = M064.pred.pred;
[~, ind] = ismember(easiest.target, M064.pred.object)
M064.pred = M064.pred.score(ind, :);
M064.tau = all_taus(easiest.bm_ind, M064.model_ind)
M064.terms = oa.ontology.term(M064.pred >= M064.tau)
easiest.M064 = M064

M091 = {};
[~, M091.model_ind] = ismember('M091', config.model)
M091.fmax = easiest.fmaxs(M091.model_ind)
M091.model = config.model{M091.model_ind}
M091.pred = load([pred_dir, 'mfo/',M091.model,'.mat']);
M091.pred = M091.pred.pred;
[~, ind] = ismember(easiest.target, M091.pred.object);
M091.pred = M091.pred.score(ind, :);
M091.tau = all_taus(easiest.bm_ind, M091.model_ind)
M091.terms = oa.ontology.term(M091.pred >= M091.tau)
easiest.M091 = M091

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = easiest.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load([pred_dir, 'mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(easiest.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(easiest.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
easiest.M082 = M082

M028 = {};
[~, M028.model_ind] = ismember('M028', config.model)
M028.fmax = easiest.fmaxs(M028.model_ind)
M028.model = config.model{M028.model_ind}
M028.pred = load([pred_dir, 'mfo/',M028.model,'.mat']);
M028.pred = M028.pred.pred;
[~, ind] = ismember(easiest.target, M028.pred.object);
M028.pred = M028.pred.score(ind, :);
M028.tau = all_taus(easiest.bm_ind, M028.model_ind)
M028.terms = oa.ontology.term(M028.pred >= M028.tau)
easiest.M028 = M028

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
% worst.pred = load([pred_dir, 'mfo/',worst.model,'.mat']);
% worst.pred = worst.pred.pred;
% [~, ind] = ismember(easy_exblast.target, worst.pred.object)
% worst.pred = worst.pred.score(ind, :);
% worst.tau = all_taus(easy_exblast.bm_ind, worst.model_ind)
% worst.terms = oa.ontology.term(worst.pred >= worst.tau)
% easy_exblast.worst = worst

best = {};
[best.fmax, best.model_ind] = max(easy_exblast.fmaxs)
best.model = config.model{best.model_ind}
best.pred = load([pred_dir, 'mfo/',best.model,'.mat']);
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
naive.pred = load([pred_dir, 'mfo/',naive.model,'.mat']);
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
blast.pred = load([pred_dir, 'mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(easy_exblast.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(easy_exblast.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
easy_exblast.blast = blast;

M064 = {};
[~, M064.model_ind] = ismember('M064', config.model)
M064.fmax = easy_exblast.fmaxs(M064.model_ind)
M064.model = config.model{M064.model_ind}
M064.pred = load([pred_dir, 'mfo/',M064.model,'.mat']);
M064.pred = M064.pred.pred;
[~, ind] = ismember(easy_exblast.target, M064.pred.object)
M064.pred = M064.pred.score(ind, :);
M064.tau = all_taus(easy_exblast.bm_ind, M064.model_ind)
M064.terms = oa.ontology.term(M064.pred >= M064.tau)
easy_exblast.M064 = M064

M091 = {};
[~, M091.model_ind] = ismember('M091', config.model)
M091.fmax = easy_exblast.fmaxs(M091.model_ind)
M091.model = config.model{M091.model_ind}
M091.pred = load([pred_dir, 'mfo/',M091.model,'.mat']);
M091.pred = M091.pred.pred;
[~, ind] = ismember(easy_exblast.target, M091.pred.object);
M091.pred = M091.pred.score(ind, :);
M091.tau = all_taus(easy_exblast.bm_ind, M091.model_ind)
M091.terms = oa.ontology.term(M091.pred >= M091.tau)
easy_exblast.M091 = M091

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = easy_exblast.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load([pred_dir, 'mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(easy_exblast.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(easy_exblast.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
easy_exblast.M082 = M082

M028 = {};
[~, M028.model_ind] = ismember('M028', config.model)
M028.fmax = easy_exblast.fmaxs(M028.model_ind)
M028.model = config.model{M028.model_ind}
M028.pred = load([pred_dir, 'mfo/',M028.model,'.mat']);
M028.pred = M028.pred.pred;
[~, ind] = ismember(easy_exblast.target, M028.pred.object);
M028.pred = M028.pred.score(ind, :);
M028.tau = all_taus(easy_exblast.bm_ind, M028.model_ind)
M028.terms = oa.ontology.term(M028.pred >= M028.tau)
easy_exblast.M028 = M028

%%
[~, hardest.target_ind] = ismember(hardest.target, oa.object)
hardest.truth.terms = oa.ontology.term(oa.annotation(hardest.target_ind, :))

hardest.fmaxs = all_fmaxs(hardest.bm_ind, :);
hardest.taus = all_taus(hardest.bm_ind, :);

best = {};
[best.fmax, best.model_ind] = max(hardest.fmaxs)
best.model = config.model{best.model_ind}
best.pred = load([pred_dir, 'mfo/',best.model,'.mat']);
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
% naive.pred = load([pred_dir, 'mfo/',naive.model,'.mat']);
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
blast.pred = load([pred_dir, 'mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(hardest.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(hardest.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
hardest.blast = blast
% hardest.blast = get_model_prediction_results('BB4S', hardest, config);

M064 = {};
[~, M064.model_ind] = ismember('M064', config.model)
M064.fmax = hardest.fmaxs(M064.model_ind)
M064.model = config.model{M064.model_ind}
M064.pred = load([pred_dir, 'mfo/',M064.model,'.mat']);
M064.pred = M064.pred.pred;
[~, ind] = ismember(hardest.target, M064.pred.object)
M064.pred = M064.pred.score(ind, :);
M064.tau = all_taus(hardest.bm_ind, M064.model_ind)
M064.terms = oa.ontology.term(M064.pred >= M064.tau)
hardest.M064 = M064

M091 = {};
[~, M091.model_ind] = ismember('M091', config.model)
M091.fmax = hardest.fmaxs(M091.model_ind)
M091.model = config.model{M091.model_ind}
M091.pred = load([pred_dir, 'mfo/',M091.model,'.mat']);
M091.pred = M091.pred.pred;
[~, ind] = ismember(hardest.target, M091.pred.object)
M091.pred = M091.pred.score(ind, :);
M091.tau = all_taus(hardest.bm_ind, M091.model_ind)
M091.terms = oa.ontology.term(M091.pred >= M091.tau)
hardest.M091 = M091

M028 = {};
[~, M028.model_ind] = ismember('M028', config.model)
M028.fmax = hardest.fmaxs(M028.model_ind)
M028.model = config.model{M028.model_ind}
M028.pred = load([pred_dir, 'mfo/',M028.model,'.mat']);
M028.pred = M028.pred.pred;
[~, ind] = ismember(hardest.target, M028.pred.object)
M028.pred = M028.pred.score(ind, :);
M028.tau = all_taus(hardest.bm_ind, M028.model_ind)
M028.terms = oa.ontology.term(M028.pred >= M028.tau)
hardest.M028 = M028

%%
[~, medium.target_ind] = ismember(medium.target, oa.object)
medium.truth.terms = oa.ontology.term(oa.annotation(medium.target_ind, :))

medium.fmaxs = all_fmaxs(medium.bm_ind, :);
medium.fmaxs(medium.fmaxs==0) = NaN;

worst = {};
[worst.fmax, worst.model_ind] = min(medium.fmaxs)
worst.model = config.model{worst.model_ind}
worst.pred = load([pred_dir, 'mfo/',worst.model,'.mat']);
worst.pred = worst.pred.pred;
[~, ind] = ismember(medium.target, worst.pred.object)
worst.pred = worst.pred.score(ind, :);
worst.tau = all_taus(medium.bm_ind, worst.model_ind)
worst.terms = oa.ontology.term(worst.pred >= worst.tau)
medium.worst = worst

naive = {};
[~, naive.model_ind] = ismember('BN4S', config.model)
naive.fmax = medium.fmaxs(naive.model_ind)
naive.model = config.model{naive.model_ind}
naive.pred = load([pred_dir, 'mfo/',naive.model,'.mat']);
naive.pred = naive.pred.pred;
[~, ind] = ismember(medium.target, naive.pred.object)
naive.pred = naive.pred.score(ind, :);
naive.tau = all_taus(medium.bm_ind, naive.model_ind)
naive.terms = oa.ontology.term(naive.pred >= naive.tau)
medium.naive = naive;

blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
blast.fmax = medium.fmaxs(blast.model_ind)
blast.model = config.model{blast.model_ind}
blast.pred = load([pred_dir, 'mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(medium.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(medium.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
medium.blast = blast;

M064 = {};
[~, M064.model_ind] = ismember('M064', config.model)
M064.fmax = medium.fmaxs(M064.model_ind)
M064.model = config.model{M064.model_ind}
M064.pred = load([pred_dir, 'mfo/',M064.model,'.mat']);
M064.pred = M064.pred.pred;
[~, ind] = ismember(medium.target, M064.pred.object)
M064.pred = M064.pred.score(ind, :);
M064.tau = all_taus(medium.bm_ind, M064.model_ind)
M064.terms = oa.ontology.term(M064.pred >= M064.tau)
medium.M064 = M064

M091 = {};
[~, M091.model_ind] = ismember('M091', config.model)
M091.fmax = medium.fmaxs(M091.model_ind)
M091.model = config.model{M091.model_ind}
M091.pred = load([pred_dir, 'mfo/',M091.model,'.mat']);
M091.pred = M091.pred.pred;
[~, ind] = ismember(medium.target, M091.pred.object);
M091.pred = M091.pred.score(ind, :);
M091.tau = all_taus(medium.bm_ind, M091.model_ind)
M091.terms = oa.ontology.term(M091.pred >= M091.tau)
medium.M091 = M091

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = medium.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load([pred_dir, 'mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(medium.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(medium.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
medium.M082 = M082

M028 = {};
[~, M028.model_ind] = ismember('M028', config.model)
M028.fmax = medium.fmaxs(M028.model_ind)
M028.model = config.model{M028.model_ind}
M028.pred = load([pred_dir, 'mfo/',M028.model,'.mat']);
M028.pred = M028.pred.pred;
[~, ind] = ismember(medium.target, M028.pred.object);
M028.pred = M028.pred.score(ind, :);
M028.tau = all_taus(medium.bm_ind, M028.model_ind)
M028.terms = oa.ontology.term(M028.pred >= M028.tau)
medium.M028 = M028


%%
[~, medium1.target_ind] = ismember(medium1.target, oa.object)
medium1.truth.terms = oa.ontology.term(oa.annotation(medium1.target_ind, :))

medium1.fmaxs = all_fmaxs(medium1.bm_ind, :);
medium1.fmaxs(medium1.fmaxs==0) = NaN;

worst = {};
[worst.fmax, worst.model_ind] = min(medium1.fmaxs)
worst.model = config.model{worst.model_ind}
worst.pred = load([pred_dir, 'mfo/',worst.model,'.mat']);
worst.pred = worst.pred.pred;
[~, ind] = ismember(medium1.target, worst.pred.object)
worst.pred = worst.pred.score(ind, :);
worst.tau = all_taus(medium1.bm_ind, worst.model_ind)
worst.terms = oa.ontology.term(worst.pred >= worst.tau)
medium1.worst = worst

naive = {};
[~, naive.model_ind] = ismember('BN4S', config.model)
naive.fmax = medium1.fmaxs(naive.model_ind)
naive.model = config.model{naive.model_ind}
naive.pred = load([pred_dir, 'mfo/',naive.model,'.mat']);
naive.pred = naive.pred.pred;
[~, ind] = ismember(medium1.target, naive.pred.object)
naive.pred = naive.pred.score(ind, :);
naive.tau = all_taus(medium1.bm_ind, naive.model_ind)
naive.terms = oa.ontology.term(naive.pred >= naive.tau)
medium1.naive = naive;

blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
blast.fmax = medium1.fmaxs(blast.model_ind)
blast.model = config.model{blast.model_ind}
blast.pred = load([pred_dir, 'mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(medium1.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(medium1.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
medium1.blast = blast;

M064 = {};
[~, M064.model_ind] = ismember('M064', config.model)
M064.fmax = medium1.fmaxs(M064.model_ind)
M064.model = config.model{M064.model_ind}
M064.pred = load([pred_dir, 'mfo/',M064.model,'.mat']);
M064.pred = M064.pred.pred;
[~, ind] = ismember(medium1.target, M064.pred.object)
M064.pred = M064.pred.score(ind, :);
M064.tau = all_taus(medium1.bm_ind, M064.model_ind)
M064.terms = oa.ontology.term(M064.pred >= M064.tau)
medium1.M064 = M064

M091 = {};
[~, M091.model_ind] = ismember('M091', config.model)
M091.fmax = medium1.fmaxs(M091.model_ind)
M091.model = config.model{M091.model_ind}
M091.pred = load([pred_dir, 'mfo/',M091.model,'.mat']);
M091.pred = M091.pred.pred;
[~, ind] = ismember(medium1.target, M091.pred.object);
M091.pred = M091.pred.score(ind, :);
M091.tau = all_taus(medium1.bm_ind, M091.model_ind)
M091.terms = oa.ontology.term(M091.pred >= M091.tau)
medium1.M091 = M091

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = medium1.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load([pred_dir, 'mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(medium1.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(medium1.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
medium1.M082 = M082

M028 = {};
[~, M028.model_ind] = ismember('M028', config.model)
M028.fmax = medium1.fmaxs(M028.model_ind)
M028.model = config.model{M028.model_ind}
M028.pred = load([pred_dir, 'mfo/',M028.model,'.mat']);
M028.pred = M028.pred.pred;
[~, ind] = ismember(medium1.target, M028.pred.object);
M028.pred = M028.pred.score(ind, :);
M028.tau = all_taus(medium1.bm_ind, M028.model_ind)
M028.terms = oa.ontology.term(M028.pred >= M028.tau)
medium1.M028 = M028


%%
[~, medium2.target_ind] = ismember(medium2.target, oa.object)
medium2.truth.terms = oa.ontology.term(oa.annotation(medium2.target_ind, :))

medium2.fmaxs = all_fmaxs(medium2.bm_ind, :);
medium2.fmaxs(medium2.fmaxs==0) = NaN;

worst = {};
[worst.fmax, worst.model_ind] = min(medium2.fmaxs)
worst.model = config.model{worst.model_ind}
worst.pred = load([pred_dir, 'mfo/',worst.model,'.mat']);
worst.pred = worst.pred.pred;
[~, ind] = ismember(medium2.target, worst.pred.object)
worst.pred = worst.pred.score(ind, :);
worst.tau = all_taus(medium2.bm_ind, worst.model_ind)
worst.terms = oa.ontology.term(worst.pred >= worst.tau)
medium2.worst = worst

naive = {};
[~, naive.model_ind] = ismember('BN4S', config.model)
naive.fmax = medium2.fmaxs(naive.model_ind)
naive.model = config.model{naive.model_ind}
naive.pred = load([pred_dir, 'mfo/',naive.model,'.mat']);
naive.pred = naive.pred.pred;
[~, ind] = ismember(medium2.target, naive.pred.object)
naive.pred = naive.pred.score(ind, :);
naive.tau = all_taus(medium2.bm_ind, naive.model_ind)
naive.terms = oa.ontology.term(naive.pred >= naive.tau)
medium2.naive = naive;

blast = {};
[~, blast.model_ind] = ismember('BB4S', config.model)
blast.fmax = medium2.fmaxs(blast.model_ind)
blast.model = config.model{blast.model_ind}
blast.pred = load([pred_dir, 'mfo/',blast.model,'.mat']);
blast.pred = blast.pred.pred;
[~, ind] = ismember(medium2.target, blast.pred.object)
blast.pred = blast.pred.score(ind, :);
blast.tau = all_taus(medium2.bm_ind, blast.model_ind)
blast.terms = oa.ontology.term(blast.pred >= blast.tau)
medium2.blast = blast;

M064 = {};
[~, M064.model_ind] = ismember('M064', config.model)
M064.fmax = medium2.fmaxs(M064.model_ind)
M064.model = config.model{M064.model_ind}
M064.pred = load([pred_dir, 'mfo/',M064.model,'.mat']);
M064.pred = M064.pred.pred;
[~, ind] = ismember(medium2.target, M064.pred.object)
M064.pred = M064.pred.score(ind, :);
M064.tau = all_taus(medium2.bm_ind, M064.model_ind)
M064.terms = oa.ontology.term(M064.pred >= M064.tau)
medium2.M064 = M064

M091 = {};
[~, M091.model_ind] = ismember('M091', config.model)
M091.fmax = medium2.fmaxs(M091.model_ind)
M091.model = config.model{M091.model_ind}
M091.pred = load([pred_dir, 'mfo/',M091.model,'.mat']);
M091.pred = M091.pred.pred;
[~, ind] = ismember(medium2.target, M091.pred.object);
M091.pred = M091.pred.score(ind, :);
M091.tau = all_taus(medium2.bm_ind, M091.model_ind)
M091.terms = oa.ontology.term(M091.pred >= M091.tau)
medium2.M091 = M091

M082 = {};
[~, M082.model_ind] = ismember('M082', config.model)
M082.fmax = medium2.fmaxs(M082.model_ind)
M082.model = config.model{M082.model_ind}
M082.pred = load([pred_dir, 'mfo/',M082.model,'.mat']);
M082.pred = M082.pred.pred;
[~, ind] = ismember(medium2.target, M082.pred.object);
M082.pred = M082.pred.score(ind, :);
M082.tau = all_taus(medium2.bm_ind, M082.model_ind)
M082.terms = oa.ontology.term(M082.pred >= M082.tau)
medium2.M082 = M082

M028 = {};
[~, M028.model_ind] = ismember('M028', config.model)
M028.fmax = medium2.fmaxs(M028.model_ind)
M028.model = config.model{M028.model_ind}
M028.pred = load([pred_dir, 'mfo/',M028.model,'.mat']);
M028.pred = M028.pred.pred;
[~, ind] = ismember(medium2.target, M028.pred.object);
M028.pred = M028.pred.score(ind, :);
M028.tau = all_taus(medium2.bm_ind, M028.model_ind)
M028.terms = oa.ontology.term(M028.pred >= M028.tau)
medium2.M028 = M028



