from graphviz import Digraph

class OntVisual:
    def __init__(self, ont, title, prefix="", data=None):
        dot = Digraph(comment=title)
        self.dot = dot
        for tid, term in ont.terms.items():
            nid = term.id.replace(':','')
            # if margin is not None:
            #     dot.node(nid, term.name, margin=margin)
            # else:
            node_text = prefix+nid+'\n%s'%term.name
            if data:
                if tid in data:
                    node_text += '\n%s' % data[tid]
                else:
                    continue
            print(tid, data[tid])
            dot.node(nid, node_text)
            for pid in term.parents:
                if pid not in ont.terms:
                    continue
                if data and pid not in data:
                    continue
                pid = pid.replace(':','')
                # p = dot.terms[pid]
                dot.edge(pid, nid)
    
    def render(self, filename, fmt='pdf', *args, **kwargs):
        self.dot.render(*args, filename=filename, format=fmt, view=True, **kwargs)

class PredVisual(OntVisual):
    def __init__(self, pred, truth, title):
        pass
    
    
