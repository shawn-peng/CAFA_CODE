%% Ontologies
% onts = pfp_ontbuild('~/workspace/cafa4/annotations/gene_ontology_edit.obo.2020-01-01')
onts = pfp_ontbuild('~/workspace/cafa4/annotations/gene_ontology_edit.obo.2020-01-01')
HPO = pfp_ontbuild('~/workspace/cafa4/annotations/hp.obo.2020-06-24')

gont.BPO = onts{1};
gont.CCO = onts{2};
gont.MFO = onts{3};

%% OA objects
mfoa = pfp_oabuild(gont.MFO, [bmdir, 't1/all.csv']);
bpoa = pfp_oabuild(gont.BPO, [bmdir, 't1/all.csv']);
ccoa = pfp_oabuild(gont.CCO, [bmdir, 't1/all.csv']);
hpoa = pfp_oabuild(HPO, [bmdir, 'hpo_t1/hpo.csv']);

%% Leaves

cafa_hist_oa_depth('~/workspace/cafa/figures/mfo_hist_depths_leaves.png', ...
    '',...'(Molecular Function) Histogram of Term Depth (Leaves)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/mfo_all_typex.txt', mfoa)

cafa_hist_oa_depth('~/workspace/cafa/figures/bpo_hist_depths_leaves.png', ...
    '',...'(Biological Process) Histogram of Term Depth (Leaves)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/bpo_all_typex.txt', bpoa)

cafa_hist_oa_depth('~/workspace/cafa/figures/cco_hist_depths_leaves.png', ...
    '',...'(Cellular Component) Histogram of Term Depth (Leaves)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/cco_all_typex.txt', ccoa)

cafa_hist_oa_depth('~/workspace/cafa/figures/hpo_hist_depths_leaves.png', ...
    '',...'(Cellular Component) Histogram of Term Depth (Leaves)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/hpo_all_typex.txt', hpoa)


%% Propagated

cafa_hist_oa_depth_propagated('~/workspace/cafa/figures/mfo_hist_depths_propagated.png', ...
    '',...'(Molecular Function) Histogram of Term Depth (Propagated)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/mfo_all_typex.txt', mfoa)

cafa_hist_oa_depth_propagated('~/workspace/cafa/figures/bpo_hist_depths_propagated.png', ...
    '',...'(Biological Process) Histogram of Term Depth (Propagated)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/bpo_all_typex.txt', bpoa)

cafa_hist_oa_depth_propagated('~/workspace/cafa/figures/cco_hist_depths_propagated.png', ...
    '',...'(Cellular Component) Histogram of Term Depth (Propagated)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/cco_all_typex.txt', ccoa)

cafa_hist_oa_depth_propagated('~/workspace/cafa/figures/hpo_hist_depths_propagated.png', ...
    '',...'(Cellular Component) Histogram of Term Depth (Propagated)', ...
    '~/workspace/cafa4/annotations/benchmark/lists/hpo_all_typex.txt', hpoa)

