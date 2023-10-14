
% run('~/workspace/cafa4/config.m')
% run('~/workspace/cafa1/config.m')
% run('~/workspace/cafa3/config.m')
% run('~/workspace/cafa3/config_cafa3.m')

% benchmark_dir = bmdir;


% onts = pfp_ontbuild('~/workspace/cafa4/annotations/gene_ontology_edit.obo.2020-01-01')
% onts = pfp_ontbuild('~/workspace/cafa3/annotations/gene_ontology_edit.obo.2017-01-01')
onts = pfp_ontbuild(onts_file)

gont.BPO = onts{1};
gont.CCO = onts{2};
gont.MFO = onts{3};

mkdir([bmdir, 'groundtruth']);

oa = pfp_oabuild(gont.MFO, [bmdir, 't1/all.csv']);
eia = pfp_eia(gont.MFO.DAG, oa.annotation);
save([bmdir, 'groundtruth/mfoa.mat'], 'oa', 'eia');

oa = pfp_oabuild(gont.BPO, [bmdir, 't1/all.csv']);
eia = pfp_eia(gont.BPO.DAG, oa.annotation);
save([bmdir, 'groundtruth/bpoa.mat'], 'oa', 'eia');

oa = pfp_oabuild(gont.CCO, [bmdir, 't1/all.csv']);
eia = pfp_eia(gont.CCO.DAG, oa.annotation);
save([bmdir, 'groundtruth/ccoa.mat'], 'oa', 'eia');

% writecell(oa.object, [bmdir, 'lists/mfo_all_typex.txt');

mkdir([bmdir, 'lists']);

goa0 = [bmdir, 't0/all.csv'];
goa1 = [bmdir, 't1/all.csv'];
bm = cafa_bm_build_go(gont, goa0, goa1)

writecell(bm.type1_mfo, [bmdir, 'lists/mfo_all_type1.txt'])
writecell(bm.type2_mfo, [bmdir, 'lists/mfo_all_type2.txt'])
writecell(bm.type1_bpo, [bmdir, 'lists/bpo_all_type1.txt'])
writecell(bm.type2_bpo, [bmdir, 'lists/bpo_all_type2.txt'])
writecell(bm.type1_cco, [bmdir, 'lists/cco_all_type1.txt'])
writecell(bm.type2_cco, [bmdir, 'lists/cco_all_type2.txt'])

species_list = {
};

lists_dir = [bmdir, 'lists/'];
system(['cp merge.sh ', lists_dir]);
merge_cmd = ['pushd ', lists_dir, ' && ./merge.sh && popd'];
system(merge_cmd);

mkdir([importdir, '/mfo']);
mkdir([importdir, '/bpo']);
mkdir([importdir, '/cco']);

mkdir(bootstrap_dir);
% system(['rm ', rundir, '/bootstrap/*']);
