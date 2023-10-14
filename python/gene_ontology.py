import re

class Term:
    #terms = {}
    def __init__(self, tid):
        self.id = tid
        self.parents = set()
        self.children = set()
        self.name = ""
        self.aspect = ''
        #Term.terms[tid] = self
    
    def setName(self, name):
        self.name = name
    def setAspect(self, aspect):
        self.aspect = aspect
    def addParent(self, pid):
        self.parents.add(pid)
    def addChild(self, cid):
        self.children.add(cid)

    def __repr__(self):
        return str(self.__dict__)
    
    #@classmethod
    #def getTerm(cls, tid):
    #    if tid in Term.terms:
    #        return Term.terms[tid]
    #    return None

term_idx = {}

aspect_map = {
    'biological_process': 'P',
    'molecular_function': 'F',
    'cellular_component': 'C',
}

def load_ont(tabont_file):
    tabont_file = open(tabont_file)
    terms = {}
    for line in tabont_file:
        row = line.rstrip().split('\t')
        cid = row[0]
        pid = row[2]




goterm_fullnames = {}
def parse_ont(ont_file):
#     goterm_fullnames = {}
    go_obo = open(ont_file)
    state = 'header'
    termid = None
    termname = None
    termnamespace = None

    termregex = "GO:[0-9]{7}"
    termregex = re.compile(termregex)

    term = None
    obsolete_term = False
    terms = set()

    for line in go_obo:
        line = line.rstrip()
        if state == 'header' or state == 'next':
            if line == '[Term]':
                state = 'term'
        elif state == 'term':
    #         print(line.split(": ", 1))
            if len(line) == 0:
                state = 'next'
                goterm_fullnames[termid] = "(%s) %s" % (termnamespace, termname)
                #terms.add(termid)
                if not obsolete_term:
                    print(termid, term.parents)
                    yield(term)
                term = None
                obsolete_term = False
            else:
                field, val = line.split(": ", 1)
                if field == "id":
                    termid = val
                    term = Term(termid)
                    if termid not in term_idx:
                        n = len(term_idx)
                        term_idx[termid] = n
                elif field == "name":
                    termname = val
                    term.setName(termname)
                elif field == "is_a":
    #                 target = val.split("!", 1)
                    target = val
                    targetterm, _ = target.split(" ! ")
                    if not termregex.match(targetterm):
                        print('invalid term', targetterm, "-")
#                     ontology.write('%s\t%s\t%s\n'%(termid, targetterm, field))
                    term.addParent(targetterm)

                elif field == "relationship":
                    rel, target = val.split(" ", 1)
                    if rel == "part_of":
                        targetterm, _ = target.split(" ! ")
                        if not termregex.match(targetterm):
                            print('invalid term', targetterm, "-")
#                         # not finished # term = {'termid': termid, 'targetterm': targetterm, 'rel': rel}
#                         ontology.write('%s\t%s\t%s\n'%(termid, targetterm, rel))
                        term.addParent(targetterm)

                elif field == "is_obsolete":
                    obsolete_term = True

                elif field == "namespace":
                    termnamespace = val
                    term.setAspect(aspect_map[termnamespace])
                else:
#                     print(field)
                    pass
    if term:
        yield term


