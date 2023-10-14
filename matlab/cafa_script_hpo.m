onts.HPO = pfp_ontbuild('~/workspace/cafa4/annotations/hp.obo.2020-06-24')

hpoa0 = '~/workspace/cafa4/annotations/benchmark/hpo_t0/hpo.csv';
hpoa1 = '~/workspace/cafa4/annotations/benchmark/hpo_t1/hpo.csv';
% goa0 = '~/workspace/cafa4/annotations/HPO/filtered_hpoa_FEB12_2020.csv';
% goa1 = '~/workspace/cafa4/annotations/HPO/filtered_hpoa_MAY17_2020.csv';
bm.hpo_type1 = cafa_bm_build_type1(onts.HPO, hpoa0, hpoa1)

writecell(bm.hpo_type1, '~/workspace/cafa4/annotations/benchmark/lists/hpo_HUMAN_type1.txt')

oa = pfp_oabuild(onts.HPO, [bmdir, 'hpo_t1/hpo.csv']);
eia = pfp_eia(onts.HPO.DAG, oa.annotation);
save('~/workspace/cafa4/annotations/benchmark/groundtruth/hpoa.mat', 'oa', 'eia');

%%
cafa_driver_filter(resdir, [filtereddir, '/hpo/'], [bmdir, 'lists/xxo_all_typex.txt'])
% cafa_driver_filter('~/workspace/cafa4/results/PASSED/', '~/workspace/cafa4/results_filtered2/hpo/', [bmdir, 'lists/hpo_all_typex.txt'])

%%
cafa_driver_import([filtereddir, 'hpo/'], [importdir, 'hpo/'], onts.HPO)
cafa_driver_preeval('~/workspace/cafa4/config/preeval/hpo_preeval.job')
cafa_driver_eval('~/workspace/cafa4/config/eval/regular/hpo_HUMAN_type1_mode1.job')
cafa_driver_result('~/workspace/cafa4/evaluation/hpo_HUMAN_type1_mode1/', '~/workspace/cafa4/config/register.tab', 'BN4S', 'BB4S', 'all')

