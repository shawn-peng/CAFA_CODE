
% goa0 = '~/workspace/cafa4/annotations/FEB14_2020/filtered/10090.csv';
% goa1 = '~/workspace/cafa4/annotations/MAY17_2020/filtered/10090.csv';

goa0 = '~/workspace/cafa4/annotations/benchmark/t0/all.csv';
goa1 = '~/workspace/cafa4/annotations/benchmark/t1/all.csv';
bm = cafa_bm_build_go(gont, goa0, goa1)

goa0 = '~/workspace/cafa4/annotations/FEB14_2020/filtered/10090.csv';
goa1 = '~/workspace/cafa4/annotations/MAY17_2020/filtered/10090.csv';
bm = cafa_bm_build_go(gont, goa0, goa1)


goa0 = '~/workspace/cafa4/annotations/benchmark/t0/226900.csv';
goa1 = '~/workspace/cafa4/annotations/benchmark/t1/226900.csv';
bm = cafa_bm_build_go(gont, goa0, goa1)

goa0 = '~/workspace/cafa4/annotations/benchmark/t0/3702.csv';
goa1 = '~/workspace/cafa4/annotations/benchmark/t1/3702.csv';
bm = cafa_bm_build_go(gont, goa0, goa1)