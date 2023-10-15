import pandas as pd
import Bio.SeqIO
from Bio.Seq import Seq

def getTargetIDMapping(sp_id, mappings_dir):
    target_map = pd.read_csv(mappings_dir+"mapping.%s.map"%sp_id, delimiter='\t', names=["target", "protid"])
    target_map = target_map.set_index('protid')
    target_map = target_map.to_dict()['target']
    return target_map

def getTargets(sp_id, targets_dir):
    input_db = targets_dir+"sp_species.%s.tfa"%sp_id
    targets = {}
    for rec in Bio.SeqIO.parse(input_db, 'fasta'):
        targets[rec.id] = str(rec.seq)
    return targets