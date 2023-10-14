type3_baseline_dir = [rundir, 'type3_baselines/']
mkdir(type3_baseline_dir)
mkdir([type3_baseline_dir, 'mfo/'])
mkdir([type3_baseline_dir, 'bpo/'])
mkdir([type3_baseline_dir, 'cco/'])

B = pfp_importblastp([blast_result_dir, '/sp_species.all_acc.tab']);

%% MFO
qseqid = pfp_loaditem([bmdir, '/lists/mfo_all_type3.txt'], 'char')

oa_t0_whole_mfo = pfp_oabuild(gont.MFO, t0_ann);


naive_pred = pfp_naive(qseqid, oa_t0_whole_mfo)
pfp_savevar([type3_baseline_dir, 'mfo/BN4S.mat'], naive_pred, 'pred')

blast_pred = pfp_blast(qseqid, B, oa_t0_whole_mfo)
pfp_savevar([type3_baseline_dir, 'mfo/BB4S.mat'], blast_pred, 'pred')


%% BPO
qseqid = pfp_loaditem([bmdir, 'lists/bpo_all_type3.txt'], 'char')

oa_t0_whole_bpo = pfp_oabuild(gont.BPO, t0_ann);

naive_pred = pfp_naive(qseqid, oa_t0_whole_bpo)
pfp_savevar([type3_baseline_dir, 'bpo/BN4S.mat'], naive_pred, 'pred')

blast_pred = pfp_blast(qseqid, B, oa_t0_whole_bpo)
pfp_savevar([type3_baseline_dir, 'bpo/BB4S.mat'], blast_pred, 'pred')


%% CCO
qseqid = pfp_loaditem([bmdir, 'lists/cco_all_type3.txt'], 'char')

oa_t0_whole_cco = pfp_oabuild(gont.CCO, t0_ann);

naive_pred = pfp_naive(qseqid, oa_t0_whole_cco)
pfp_savevar([type3_baseline_dir, 'cco/BN4S.mat'], naive_pred, 'pred')

blast_pred = pfp_blast(qseqid, B, oa_t0_whole_cco)
pfp_savevar([type3_baseline_dir, 'cco/BB4S.mat'], blast_pred, 'pred')



