import pcsgraphs.classes as g


def load(path):
    graph = g.DirectedGraph()
    graph.add_from_files(path)
