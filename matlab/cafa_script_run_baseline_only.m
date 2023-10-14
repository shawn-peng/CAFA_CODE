
% run('~/workspace/cafa2/config_cafa3.m')
run('~/workspace/cafa1/config_cafa4.m')
% run('~/workspace/cafa3/config_cafa4.m')
% run('~/workspace/cafa4/config_cafa4.m')

% cafa_script_init
cafa_script_init2

cafa_script_baselines

%%
cafa_driver_filter(resdir, [filtereddir, '/all_ont/'], [bmdir, 'lists/xxo_all_typex.txt'])

%%
cafa_driver_import([filtereddir, 'all_ont/'], [importdir, 'mfo/'], gont.MFO)
cafa_driver_import([filtereddir, 'all_ont/'], [importdir, 'bpo/'], gont.BPO)
cafa_driver_import([filtereddir, 'all_ont/'], [importdir, 'cco/'], gont.CCO)

%%
cafa_driver_preeval([cfgdir, 'preeval/mfo_preeval.job'])
cafa_driver_preeval([cfgdir, 'preeval/bpo_preeval.job'])
cafa_driver_preeval([cfgdir, 'preeval/cco_preeval.job'])

%%
cafa_driver_eval([cfgdir, 'eval/regular/mfo_all_type1_mode1.job'])
cafa_driver_eval([cfgdir, 'eval/regular/bpo_all_type1_mode1.job'])
cafa_driver_eval([cfgdir, 'eval/regular/cco_all_type1_mode1.job'])

cafa_driver_eval([cfgdir, 'eval/regular/mfo_all_type2_mode1.job'])
cafa_driver_eval([cfgdir, 'eval/regular/bpo_all_type2_mode1.job'])
cafa_driver_eval([cfgdir, 'eval/regular/cco_all_type2_mode1.job'])

%%
cafa_driver_result([evaldir, 'mfo_all_type1_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')
cafa_driver_result([evaldir, 'bpo_all_type1_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')
cafa_driver_result([evaldir, 'cco_all_type1_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')

cafa_driver_result([evaldir, 'mfo_all_type2_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')
cafa_driver_result([evaldir, 'bpo_all_type2_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')
cafa_driver_result([evaldir, 'cco_all_type2_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')

%%
% cafa_driver_eval([cfgdir, 'eval/regular/mfo_MOUSE_type1_mode1.job'])
% cafa_driver_result([evaldir, 'mfo_MOUSE_type1_mode1/'], [cfgdir, 'register.tab'], 'BN4S', 'BB4S', 'all')


