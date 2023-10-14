
%% MFO
qseqid = pfp_loaditem([bmdir, '/lists/mfo_all_typex.txt'], 'char')
if baseline_with_type3_targets
    qseqid_type3 = pfp_loaditem([type3_list_dir, 'mfo_all_type3.txt'], 'char');
    qseqid = [qseqid; qseqid_type3];
    qseqid = sort(qseqid);
end

oa_t0_whole_mfo = pfp_oabuild(gont.MFO, t0_ann);


naive_pred = pfp_naive(qseqid, oa_t0_whole_mfo)
pfp_savevar([importdir, 'mfo/BN4S.mat'], naive_pred, 'pred')

B = pfp_importblastp([blast_result_dir, '/sp_species.all_acc.tab']);
blast_pred = pfp_blast(qseqid, B, oa_t0_whole_mfo)
pfp_savevar([importdir, 'mfo/BB4S.mat'], blast_pred, 'pred')


%% BPO
qseqid = pfp_loaditem([bmdir, 'lists/bpo_all_typex.txt'], 'char')
if baseline_with_type3_targets
    qseqid_type3 = pfp_loaditem([type3_list_dir, 'bpo_all_type3.txt'], 'char');
    qseqid = [qseqid; qseqid_type3];
    qseqid = sort(qseqid);
end

oa_t0_whole_bpo = pfp_oabuild(gont.BPO, t0_ann);


naive_pred = pfp_naive(qseqid, oa_t0_whole_bpo)
pfp_savevar([importdir, 'bpo/BN4S.mat'], naive_pred, 'pred')

B = pfp_importblastp([blast_result_dir, '/sp_species.all_acc.tab']);
blast_pred = pfp_blast(qseqid, B, oa_t0_whole_bpo)
pfp_savevar([importdir, 'bpo/BB4S.mat'], blast_pred, 'pred')


%% CCO
qseqid = pfp_loaditem([bmdir, 'lists/cco_all_typex.txt'], 'char')
if baseline_with_type3_targets
    qseqid_type3 = pfp_loaditem([type3_list_dir, 'cco_all_type3.txt'], 'char');
    qseqid = [qseqid; qseqid_type3];
    qseqid = sort(qseqid);
end

oa_t0_whole_cco = pfp_oabuild(gont.CCO, t0_ann);

naive_pred = pfp_naive(qseqid, oa_t0_whole_cco)
pfp_savevar([importdir, 'cco/BN4S.mat'], naive_pred, 'pred')

B = pfp_importblastp([blast_result_dir, '/sp_species.all_acc.tab']);
blast_pred = pfp_blast(qseqid, B, oa_t0_whole_cco)
pfp_savevar([importdir, 'cco/BB4S.mat'], blast_pred, 'pred')


%% HPO
HPO = pfp_ontbuild('~/workspace/cafa4/annotations/hp.obo.2020-06-24')
qseqid = pfp_loaditem([bmdir, 'lists/hpo_all_typex.txt'], 'char')

% oa_t0_whole_hpo = pfp_oabuild(HPO, ['~/workspace/cafa4/annotations/hpoa_t0.csv']);
oa_t0_whole_hpo = pfp_oabuild(HPO, t0_ann_hpo);

naive_pred = pfp_naive(qseqid, oa_t0_whole_hpo)
pfp_savevar([importdir, 'hpo/BN4S.mat'], naive_pred, 'pred')

B = pfp_importblastp([blast_species_result_dir, '/sp_species.9606_acc.tab']);
blast_pred = pfp_blast(qseqid, B, oa_t0_whole_hpo)
pfp_savevar([importdir, 'hpo/BB4S.mat'], blast_pred, 'pred')




