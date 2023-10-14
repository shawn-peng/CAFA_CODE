
resdir = '~/workspace/cafa4/results/PASSED'
filtereddir = '~/workspace/cafa4/results_filtered2/'
importdir = '~/workspace/cafa4/imported/'
bmdir = '~/workspace/cafa4/annotations/benchmark/'

species_list = {
    '10090';
    '10116';
    '3702';
    '44689';
    '559292';
    '7227';
    '7955';
    '9606';
    '9823';
};

n = size(species_list, 1);

for i = 1:n
    species = species_list{i}
    bm = [bmdir, species, '.csv'];
    speciesdir = [filtereddir, species, '/'];
    outdir = [importdir, species, '/'];
    for 
    cafa_driver_import(speciesdir, outdir, )
end
