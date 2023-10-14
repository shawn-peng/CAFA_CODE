import sys

import progressbar

from annotation import *
from ontology import *
from gene_ontology import *
from mat_pred import *
from mat_ontology import *
from graph import *

import pandas as pd
from collections import defaultdict
from sortedcontainers import SortedList
from itertools import starmap
import heapq
from heapq import heappush, heappop
import random
import numpy as np
import matplotlib.pyplot as plt
import traceback

import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle

import glob
import os
import traceback

asp_map = {'mfo': 'F', 'bpo': 'P', 'cco': 'C'}


def get_anns(f, ont, aspect="FPC"):
    f = open(f)
    leaf_anns = defaultdict(set)
    full_anns = {}
    for l in f:
        # print(l)
        p, t, a = l.rstrip().split('\t')
        if a in aspect:
            leaf_anns[p].add(t)
    for target, leaves in leaf_anns.items():
        lann = LeafAnnotation(leaves)
        full_anns[target] = FullAnnotation(ont, lann.terms)
    return full_anns


def buildbm(bm, ont_name, ont, t0_anns_asp):
    asp = asp_map[ont_name]
    t0_anns = t0_anns_asp[ont_name]
    bm = open(bm)
    ann_table = defaultdict(set)
    for l in bm:
        p, t, a = l.rstrip().split('\t')
        #         print(p, t, a)
        if p not in t0_anns:
            continue
        if a != asp:
            continue
        ann_table[p].add(t)

    bm = {}
    for p, terms in ann_table.items():
        ann = ont.getSubgraph(terms)
        if list(ann.keys()) == ['GO:0005515']:
            continue
        bm[p] = ann
    return bm


def information_content(ont, ann_t0):
    pos = defaultdict(lambda: 1)
    neg = defaultdict(int)

    i = 0
    with ProgressBar(max_value=len(ont.keys())) as bar:
        for tid in ont.keys():
            bar.update(i)
            i += 1
            term = ont[tid]
            for p, ann in ann_t0.items():
                if tid in ann:
                    pos[tid] += 1
                else:
                    has_all_parents = True
                    for ptid in term.parents:
                        if ptid not in ann:
                            has_all_parents = False
                            break
                    if has_all_parents:
                        neg[tid] += 1
            # print(tid)
            # print('pos:', pos[tid], 'neg:', neg[tid])
    ic = {}
    for tid in ont.keys():
        if not pos[tid] + neg[tid]:
            ic[tid] = np.nan
        else:
            ic[tid] = -np.log2(pos[tid] / (pos[tid] + neg[tid]))

    return ic


def ont_diff(args):
    ont, ont2 = args
    ret = ont - ont2
    print(len(ret.terms))
    return list(ret.terms.keys())


def get_eval_onts(bmdict, t0_anns_asp, cafa4_ont_asp, ont_names=['mfo', 'bpo', 'cco']):
    jobs = defaultdict(list)
    for ont_name in ont_names:
        bm = bmdict[ont_name]
        t0_anns = t0_anns_asp[ont_name]

        ont_jobs = []
        jobs[ont_name] = ont_jobs
        print(ont_name)
        i = 0
        for p in bm.keys():
            if p not in t0_anns:
                print(p)
                continue
            ont_jobs.append((cafa4_ont_asp[ont_name], t0_anns[p]))

    eval_onts_asp = {}

    for ont_name in ['mfo', 'bpo', 'cco']:
        with Pool(32) as p:
            res = p.map(ont_diff, jobs[ont_name])

        bm = bmdict[ont_name]
        eval_onts = {}
        eval_onts_asp[ont_name] = eval_onts
        for p, eval_ont in zip(bm.keys(), res):
            eval_onts[p] = eval_ont
    # pickle.dump(eval_onts_asp, open('type3_eval_onts_asp.pickle', 'wb'))
    # eval_onts_asp = pickle.load(open('type3_eval_onts_asp.pickle', 'rb'))
    return eval_onts_asp


def calc_prrc(pred, truth, ont):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for t in pred:
        if t not in ont.terms:
            continue
        if t in truth.terms:
            tp += 1
        else:
            fp += 1
    for t in truth.terms:
        if t in pred:
            continue
        fn += 1

    if tp + fp == 0:
        pr = 1
    else:
        pr = tp / (tp + fp)
    rc = tp / (tp + fn)

    return pr, rc


def minmax(it):
    min = max = None
    for val in it:
        if min is None or val < min:
            min = val
        if max is None or val > max:
            max = val
    return min, max


def load_prediction(modelfile, bm=None, bm_onts=None, logfile=None):
    mf = open(modelfile)
    pred = {}
    # n = 10000000
    for line in mf:
        # if n <= 0:
        #     break
        # n -= 1
        p, t, s = line.rstrip().split('\t')
        if p not in pred:
            pred[p] = {}
        pred[p][t] = float(s)
    print('predictions for ', modelfile, 'loaded')
    return pred


def load_mat_prediction(modelfile, bm=None, bm_onts=None, logfile=None):
    pred = MatPred.read_mat_pred(modelfile)
    print('predictions for ', modelfile, 'loaded')
    return pred


def filter_scores(row, bm_ont):
    dlist = []
    for t, s in row.items():
        if t not in bm_ont:
            dlist.append(t)
    for t in dlist:
        del row[t]


def normalize_scores(row):
    smin, smax = minmax(row.values())

    def normalize(s):
        if smax == smin:
            return 1.0
        else:
            return (s - smin) / (smax - smin)

    for t, s in row.items():
        row[t] = normalize(s)

log_dir = 'type3_eval_as_whole_log/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


def eval_modelfile(modelfile, bm, bm_onts, ont, term_ic, ont_name, log_dir='type3_eval_log'):
    basename = os.path.basename(modelfile)
    modelname = os.path.splitext(basename)[0]
    print(ont_name, modelfile, len(bm))

    prrc_curves = []

    pred = load_mat_prediction(modelfile, bm, bm_onts)
    return eval_model(pred, bm, bm_onts, ont, term_ic, ont_name, modelname, log_dir=log_dir)


def eval_modelfile_macro(modelfile, bm, bm_onts, ont, term_ic, ont_name, modelname, repeats=None, log_dir='type3_eval_log', show_progress=False):
    # basename = os.path.basename(modelfile)
    # modelname = os.path.splitext(basename)[0]
    print(ont_name, modelfile, len(bm))

    prrc_curves = []

    try:
        pred = load_mat_prediction(modelfile, bm, bm_onts)
        return eval_model_macro(pred, bm, bm_onts, ont, term_ic, ont_name, modelname, log_dir=log_dir, show_progress=show_progress)
    except Exception as e:
        print('exception happend for modelfile', modelfile)
        traceback.print_exc()


def eval_target_pred(pred, groundtruth, bm_ont, term_ic, thresholds):
    res = defaultdict(lambda: np.zeros_like(thresholds))
    # q = SortedList()
    q = []

    tp = 0
    fp = 0
    tn = len(bm_ont) - len(groundtruth.terms)
    fn = len(groundtruth.terms)

    ru = 0
    mi = 0

    # add all truth terms to ru
    for t in groundtruth.terms:
        ru += term_ic[t]

    # start thres 0, all pred included
    for t, s in pred.items():
        if t not in bm_ont:
            continue
        ic = term_ic[t]
        # ru += ic
        correct = t in groundtruth
        heappush(q, (s, t, correct, ic))
        if correct:
            tp += 1
            fn -= 1
            ru -= ic  # remove from ru
        else:
            fp += 1
            tn -= 1
            mi += ic  # add to mi

    # print(tp, fp, tn, fn, ru, mi)
    n = len(thresholds)
    m = len(q)

    if m == 0:
        return None
    # print(q)

    i = 0
    while i < n:
        thres = thresholds[i]
        # print(thres)
        while q and q[0][0] < thres:
            s, t, correct, ic = heappop(q)
            # print(s, t)
            if correct:  # correct pred removed
                tp -= 1
                fn += 1
                ru += ic
            else:  # wrong pred removed
                fp -= 1
                tn += 1
                mi -= ic
        res['tp'][i] = tp
        res['fp'][i] = fp
        res['tn'][i] = tn
        res['fn'][i] = fn
        res['ru'][i] = ru
        res['mi'][i] = mi

        i += 1

    return res


# def remove_useless_points(curve):
#     print(curve)
#     res = []
#
#     if np.all(np.diff(curve[:, 1]) >= 0):
#         y_increasing = True
#         print('y increasing')
#     elif np.all(np.diff(curve[:, 1]) <= 0):
#         y_increasing = False
#         print('y decreasing')
#     else:
#         assert 0
#
#     if y_increasing:
#         for x, y in curve:
#             while res and x >= res[0][-1]:
#                 res.pop()
#             res.append((x, y))
#     else:
#         for x, y in curve:
#             if res and x <= res[0][-1]:
#                 continue
#             res.append((x, y))
#     print(res)
#     return np.array(res)
def remove_useless_points(curve):
    """ Remove points with both x, y lower than another point """
    res = []

    def useless(i, x, y):
        for ii, (xx, yy) in enumerate(curve):
            if ii == i:
                continue
            if x <= xx and y <= yy:
                return True
        return False

    for i, (x, y) in enumerate(curve):
        if useless(i, x, y):
            continue
        res.append((x, y))
    return np.array(res)


def preprocess_target_pred(pred, eval_ont):
    res = {}
    for t, score in pred.items():
        pass


def preprocess_model(model):
    for pred_t, row in model.items():
        pass


def eval_model_macro(pred, bm, bm_onts, ont, term_ic, ont_name, modelname, repeats=None, log_dir='type3_eval_log', show_progress=False):
    logfile = os.path.join(log_dir, '%s_%s.log' % (ont_name, modelname))
    print('logging into', logfile)
    logfile = open(logfile, 'w', buffering=1)

    covered_cnt = 0
    covered_set = set()

    sorted_scores = []

    prrc_curves = []
    rumi_curves = []

    if show_progress:
        bar = progressbar.ProgressBar(max_value=len(pred))
        bar.start()
        i = 0
    # n = 1000

    # for pred_t, row in pred.items():
    for pred_t in bm.keys():
        if show_progress:
            i += 1
            bar.update(i)

        if pred_t not in bm:
            continue
        if pred_t not in bm_onts:
            continue

        logfile.write("%s\t%s\t%s\n" % (ont_name, modelname, pred_t))

        truth_ann = bm[pred_t]
        if pred_t in pred:
            row = pred[pred_t]
        else:
            row = {}
        bm_ont = set(bm_onts[pred_t])

        logfile.write("%s\t%s\n" % (len(truth_ann.terms), len(bm_ont)))

        if not repeats:
            nrep = 1
        else:
            nrep = repeats[pred_t]

        covered_cnt += nrep

        thresholds = np.arange(0, 1.01, 0.01)
        ret = eval_target_pred(row, truth_ann, bm_ont, term_ic, thresholds)
        if ret is None:
            continue

        tp = ret['tp']
        fp = ret['fp']
        tn = ret['tn']
        fn = ret['fn']
        ru = ret['ru']
        mi = ret['mi']

        pr = tp / (tp + fp)
        pr[np.isnan(pr)] = 0
        rc = tp / (tp + fn)

        # logfile.write("%f\t%f\n"%(pr, rc))

        for ir in range(nrep):
            prrc_curves.append(np.array((pr, rc)))
            rumi_curves.append(np.array((ru, mi)))
            covered_set.add(pred_t)

        # n -= 1
        # if n <= 0:
        #     break

    if show_progress:
        bar.finish()

    # return prrc_curves, rumi_curves
    prrc_curves = np.array(prrc_curves)
    rumi_curves = np.array(rumi_curves)

    print(prrc_curves.shape)

    prrc_curve = np.mean(prrc_curves, 0).T
    rumi_curve = np.mean(rumi_curves, 0).T
    print(prrc_curve.shape)

    if not prrc_curve.shape or not rumi_curve.shape:
        print(covered_set, 'should be empty')
        return ont_name, modelname, None

    res = {
        'prrc_curves': prrc_curves,
        'rumi_curves': rumi_curves,
        'prrc_curve': prrc_curve,
        'rumi_curve': rumi_curve,
        'covered_targets': covered_set,
    }
    return ont_name, modelname, res


def eval_model(pred, bm, bm_onts, ont, term_ic, ont_name, modelname, repeats=None, log_dir='type3_eval_log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = os.path.join(log_dir, '%s_%s.log' % (ont_name, modelname))
    logfile = open(logfile, 'w', buffering=1)

    covered_cnt = 0
    covered_set = set()

    sorted_scores = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    ru = 0
    mi = 0

    # n=10
    for pred_t, row in pred.items():
        if pred_t not in bm:
            continue
        if pred_t not in bm_onts:
            continue

        logfile.write("%s\t%s\t%s\n" % (ont_name, modelname, pred_t))

        truth_ann = bm[pred_t]
        bm_ont = set(bm_onts[pred_t])

        logfile.write("%s\t%s\n" % (len(truth_ann.terms), len(bm_ont)))

        if not repeats:
            nrep = 1
        else:
            nrep = repeats[pred_t]

        fn += len(truth_ann.terms) * nrep
        tn += (len(bm_ont) - len(truth_ann.terms)) * nrep
        ru += sum(map(lambda t: term_ic[t] * nrep, truth_ann.terms))

        covered_cnt += nrep

        for t, s in row.items():
            if t not in bm_ont:
                continue
            ic = term_ic[t]
            for ri in range(nrep):
                heapq.heappush(sorted_scores, (-s, (pred_t, t, ic)))
                covered_set.add(pred_t)

        # n-=1
        # if n <= 0:
        #     break

    if sorted_scores:
        curr_score = -sorted_scores[0][0]
    else:
        curr_score = None
    prrc_curve = []
    rumi_curve = []

    logfile.write("%d\n" % (len(sorted_scores)))
    while sorted_scores:
        while sorted_scores and curr_score == -sorted_scores[0][0]:
            s, (pred_t, t, ic) = heapq.heappop(sorted_scores)
            s = -s

            # logfile.write("%f\t%s\t%s\n"%(s, pred_t, t))

            if t in bm[pred_t]:
                tp += 1
                fn -= 1
                ru -= ic
            else:
                fp += 1
                tn -= 1
                mi += ic

            # logfile.write("%d\t%d\t%d\t%d\n"%(tp, fp, tn, fn))

        if sorted_scores:
            curr_score = -sorted_scores[0][0]

        pr = tp / (tp + fp)
        rc = tp / (tp + fn)

        # logfile.write("%f\t%f\n"%(pr, rc))

        prrc_curve.append((pr, rc))
        rumi_curve.append((ru / covered_cnt, mi / covered_cnt))

    prrc_curve = np.array(prrc_curve)
    print(ont_name, modelname, prrc_curve.shape, len(covered_set))
    logfile.write("%s\t%s\t%s\t%d\n" % (ont_name, modelname, prrc_curve.shape, len(covered_set)))
    logfile.close()
    return (ont_name, modelname, {
        'prrc_curve': prrc_curve,
        'rumi_curve': rumi_curve,
        'covered_targets': covered_set,
    })


def get_bootstrap_bm_iter(bm, bm_onts, bi_row):
    bbm = {t: bm[t] for t in bi_row}
    bbm_onts = {t: bm_onts[t] for t in bi_row}
    cnts = defaultdict(int)
    for t in bi_row:
        cnts[t] += 1
    return bbm, bbm_onts, cnts


# ==================================================
def eval_bootstrap_iter_with_cache(model_cache_dir, logfile, repeats, bi_num):
    covered_cnt = 0
    covered_set = set()

    sorted_scores = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    ru = 0
    mi = 0

    def _open_pickle_cache(varname):
        pickle_file = os.path.join(model_cache_dir, '%s.pickle' % varname)
        return open(pickle_file, 'rb')

    pred_info = pickle.load(_open_pickle_cache('pred_info'))
    bm = pickle.load(_open_pickle_cache('bm'))
    bm_onts = pickle.load(_open_pickle_cache('bm_onts'))
    term_ic = pickle.load(_open_pickle_cache('term_ic'))

    # logfile = open('%s/%05d.log' % (logging_dir, bi_num), 'w', buffering=1)

    print("bootstrap %d" % bi_num)
    logfile.write("bootstrap %d\n" % bi_num)

    for pred_t, row in pred_info.items():
        if pred_t not in repeats or not repeats[pred_t]:
            continue

        truth_ann = bm[pred_t]
        bm_ont = set(bm_onts[pred_t])

        # logfile.write("%s\t%s\t%s\n"%(pred_t, len(truth_ann.terms), len(bm_ont)))

        nrep = repeats[pred_t]

        fn += len(truth_ann.terms) * nrep
        tn += (len(bm_ont) - len(truth_ann.terms)) * nrep
        ru += sum(map(lambda t: term_ic[t] * nrep, truth_ann.terms))

        covered_cnt += nrep

        for t, (s, flag) in row.items():
            if t not in bm_ont:
                continue
            ic = term_ic[t]
            for ri in range(nrep):
                heapq.heappush(sorted_scores, (-s, (pred_t, t, flag, ic, ri)))  # ri is used to break the tie in heap
                covered_set.add(pred_t)

    logfile.write("total num of scores: %d\n" % (len(sorted_scores)))
    logfile.write(f"fn: {fn}, tn: {tn}, ru: {ru}\n")

    if sorted_scores:
        curr_score = -sorted_scores[0][0]
    else:
        curr_score = None
    prrc_curve = []
    rumi_curve = []

    while sorted_scores:
        while sorted_scores and curr_score == -sorted_scores[0][0]:
            s, (pred_t, t, flag, ic, ri) = heapq.heappop(sorted_scores)
            s = -s

            # logfile.write("%f\t%s\t%s\n"%(s, pred_t, t))

            # if t in bm[pred_t]:
            if flag:
                tp += 1
                fn -= 1
                ru -= ic
            else:
                fp += 1
                tn -= 1
                mi += ic

            # logfile.write("%d\t%d\t%d\t%d\n"%(tp, fp, tn, fn))

        if sorted_scores:
            curr_score = -sorted_scores[0][0]

        pr = tp / (tp + fp)
        rc = tp / (tp + fn)

        # logfile.write("%f\t%f\n"%(pr, rc))

        prrc_curve.append((pr, rc))
        rumi_curve.append((ru / covered_cnt, mi / covered_cnt))

    prrc_curve = np.array(prrc_curve)
    rumi_curve = np.array(rumi_curve)

    # logfile.write(f"rumi_curve: {str(rumi_curve)}\n")
    # print(ont_name, modelfile, prrc_curve.shape, len(covered_set))

    total_t = 0
    for pred_t, nrep in repeats.items():
        total_t += nrep
    cover_rate = covered_cnt / total_t

    logfile.write(f"bootstrap {bi_num}:\t{prrc_curve.shape}\t{len(covered_set)}\t{covered_cnt}\t{cover_rate}\n")
    print(f"bootstrap {bi_num}:\t{prrc_curve.shape}\t{len(covered_set)}\t{covered_cnt}\t{cover_rate}\n")

    return {
        'prrc_curve': prrc_curve,
        'rumi_curve': rumi_curve,
        'covered_targets': covered_set,
        'covered_count': covered_cnt,
        'cover_rate': cover_rate,
    }


bslog_dir = 'type3_eval_log_as_whole_bootstrap/'
if not os.path.exists(bslog_dir):
    os.mkdir(bslog_dir)

results_dir = 'type3_eval_bst_results/'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

eval_cache_dir = 'type3_eval_cache/'
if not os.path.exists(eval_cache_dir):
    os.mkdir(eval_cache_dir)


# def eval_bootstrap_iter_with_cache()
def eval_model_as_whole_bootstrap_parallel(modelfile, bm, bm_onts, ont, term_ic, ont_name, bi):
    ret = []
    basename = os.path.basename(modelfile)
    modelname = os.path.splitext(basename)[0]
    print(ont_name, modelfile, len(bm))

    if not os.path.exists(bslog_dir):
        os.makedirs(bslog_dir)

    logfile = os.path.join(bslog_dir, '%s_%s.log' % (ont_name, modelname))
    print('logging into', logfile)
    logfile = open(logfile, 'w', buffering=1)

    res_pickle = os.path.join(results_dir, '%s_%s.pickle' % (ont_name, modelname))
    if os.path.exists(res_pickle):
        ret = pickle.load(open(res_pickle, 'rb'))
        return (ont_name, modelname, ret)

    print('loading', modelfile)
    logfile.write('loading predictions\n')

    pred = load_mat_prediction(modelfile, bm, bm_onts, logfile=logfile)
    logfile.write('pred targets:%d\n' % len(pred))

    pred_info = defaultdict(dict)
    for pred_t, row in pred.items():
        if pred_t not in bm:
            continue
        if pred_t not in bm_onts:
            continue

        bm_ont = set(bm_onts[pred_t])
        logfile.write("%s\t%s\t%s\n" % (ont_name, modelname, pred_t))

        for t, s in row.items():
            if t not in bm_ont:
                continue
            if t in bm[pred_t]:
                pred_info[pred_t][t] = (s, True)
            else:
                pred_info[pred_t][t] = (s, False)
    # ==================================================

    model_cache_dir = os.path.join(eval_cache_dir, '%s_%s' % (ont_name, modelname))
    if not os.path.exists(model_cache_dir):
        os.mkdir(model_cache_dir)

    def _open_pickle_cache(varname):
        pickle_file = os.path.join(model_cache_dir, '%s.pickle' % varname)
        return open(pickle_file, 'wb')

    pickle.dump(pred_info, _open_pickle_cache('pred_info'))
    pickle.dump(bm, _open_pickle_cache('bm'))
    pickle.dump(bm_onts, _open_pickle_cache('bm_onts'))
    pickle.dump(term_ic, _open_pickle_cache('term_ic'))

    iter_logging_dir = os.path.join(bslog_dir, '%s_%s' % (ont_name, modelname))
    if not os.path.exists(iter_logging_dir):
        os.mkdir(iter_logging_dir)

    i = 0
    bstjobs = []
    try:
        for (bi_num, bi_row) in enumerate(bi):
            bbm, bbm_onts, brep = get_bootstrap_bm_iter(bm, bm_onts, bi_row)
            #         bstjobs.append((pred, bbm, bbm_onts, ont, ont_name, brep, bi_num))
            #         bstjobs.append((model_cache_dir, iter_logging_dir, brep, bi_num))
            bstjobs.append((model_cache_dir, logfile, brep, bi_num))
        #         if bi_num >= 64:
        #             break

        #     with Pool(8) as p:
        # #         logfile.write('%d: %d\n'%(i, len(bi_row)))
        #         ret = p.starmap(eval_bootstrap_iter_with_cache, bstjobs)
        # #         logfile.write('%s\n'%(str(brep)))
        # #        bres = eval_model_as_whole(pred, bbm, bbm_onts, ont, ont_name, brep)

        # #         n-=1
        # #         if n <= 0:
        # #             break
        ret = list(starmap(eval_bootstrap_iter_with_cache, bstjobs))

        pickle.dump(ret, open(res_pickle, 'wb'))
    except:
        traceback.print_exc(file=logfile)

    logfile.close()

    return (ont_name, modelname, ret)


def gen_bootstraps(bm, N):
    ret = []
    pop = list(bm.keys())
    m = len(bm)
    for i in range(N):
        ret.append(random.choices(pop, k=m))
    return ret


# pi_map = None
# team_map = None

def register_pi_map(d):
    global pi_map
    pi_map = d
    print(len(pi_map))


def register_team_map(d):
    global team_map
    team_map = d


def analyze_type3_eval_single(elem):
    global pi_map, team_map
    if elem is None:
        return
    ont_name, modelname, res = elem
    modelname = modelname.split('.')[0]
    if res is None:
        return

    m_pr = []

    #     sres = {'bstres': bstres}
    sres = deepcopy(res)

    sres['pi'] = pi_map[modelname]
    sres['teamname'] = team_map[modelname]
    print(sres['pi'])

    m_f = []
    m_s = []

    prrc_curve = np.array(sres['prrc_curve'])
    rumi_curve = np.array(sres['rumi_curve'])

    prrc_curve = remove_useless_points(prrc_curve)
    rumi_curve = -remove_useless_points(-rumi_curve)
    sres['prrc_curve'] = prrc_curve
    sres['rumi_curve'] = rumi_curve

    # if len(prrc_curve.shape) == 1:  # or data.shape[0] == 1:
    #     return

    try:
        m = prrc_curve
        pr = m[:, 0]
        rc = m[:, 1]

        m = rumi_curve
        ru = m[:, 0]
        mi = m[:, 1]
        if prrc_curve.shape[0] == 1:
            sres['fmax'] = np.nan
            sres['smin'] = np.nan
            sres['fmax_tauidx'] = 0
            sres['smin_tauidx'] = 0
        else:
            sres['fmax'] = np.nanmax(2 * pr * rc / (pr + rc))
            sres['smin'] = np.nanmin((ru ** 2 + mi ** 2) ** (1 / 2))
            sres['fmax_tauidx'] = np.nanargmax(2 * pr * rc / (pr + rc))
            sres['smin_tauidx'] = np.nanargmin((ru ** 2 + mi ** 2) ** (1 / 2))
    except Exception as e:
        print('exception for', ont_name, modelname, res, e)
        sres['fmax'] = np.nan
        sres['smin'] = np.nan
        sres['fmax_tauidx'] = 0
        sres['smin_tauidx'] = 0

    print(sres['prrc_curve'])
    return ont_name, modelname, sres


def append_cover_rate(elem, bm_dict):
    if elem is None:
        return elem
    ont_name, modelname, res = elem
    modelname = modelname.split('.')[0]
    if res is None:
        print(elem)
        return elem

    res['cover_rate'] = len(res['covered_targets']) / len(bm_dict[ont_name])
    print(res['cover_rate'])
    return elem


def analyze_type3_eval_bst_single(elem):
    #     print(elem)
    if elem is None:
        return
    ont_name, modelname, bstres = elem

    m_pr = []

    #     sres = {'bstres': bstres}
    sres = {}

    sres['pi'] = pi_map[modelname]
    sres['teamname'] = team_map[modelname]
    print(sres['pi'])

    sres['cover_rate'] = np.array([ibstres['cover_rate'] for ibstres in bstres]).mean()

    # for each bootstrap
    m_f = []
    m_s = []
    #     with progressbar.ProgressBar(max_value=len(bstres)) as bar:
    for i, ibstres in enumerate(bstres):
        #         bar.update(i)
        prrc_curve = ibstres['prrc_curve']
        rumi_curve = ibstres['rumi_curve']

        if len(prrc_curve.shape) == 1:  # or data.shape[0] == 1:
            return
        m = prrc_curve
        pr = m[:, 0]
        rc = m[:, 1]

        m = rumi_curve
        ru = m[:, 0]
        mi = m[:, 1]

        if prrc_curve.shape[0] == 1:
            m_f.append(np.nan)
            m_s.append(np.nan)
            continue

        m_f.append(np.nanmax(2 * pr * rc / (pr + rc)))
        m_s.append(np.nanmin(ru ** 2 + mi ** 2) ** (1 / 2))

    sres['fmaxs'] = m_f
    sres['fmax'] = np.nanmean(m_f)
    sres['smins'] = m_s
    sres['smin'] = np.nanmean(m_s)

    fmax_q05, fmax_q95 = np.percentile(m_f, [5, 95])
    sres['fmax_q05'] = fmax_q05
    sres['fmax_q95'] = fmax_q95
    smin_q05, smin_q95 = np.percentile(m_s, [5, 95])
    sres['smin_q05'] = smin_q05
    sres['smin_q95'] = smin_q95

    #     print(sres.keys())

    print(smin_q05, smin_q95)

    #     eval_res[ont_name][modelname] = sres
    #     return m_pr
    return ont_name, modelname, sres


def analyze_type3_eval_bst_process(job):
    single_res = load_type3_eval_result_pickle(*job)
    return analyze_type3_eval_bst_single(single_res)


results_dir = 'type3_eval_bst_results/'


def load_type3_eval_result_pickle(modelfile, bm, bm_onts, ont, ont_name):
    ret = []
    basename = os.path.basename(modelfile)
    modelname = os.path.splitext(basename)[0]
    print(ont_name, modelfile, len(bm))

    res_pickle = os.path.join(results_dir, '%s_%s.pickle' % (ont_name, modelname))
    if os.path.exists(res_pickle):
        ret = pickle.load(open(res_pickle, 'rb'))
        return (ont_name, modelname, ret)

    return None
