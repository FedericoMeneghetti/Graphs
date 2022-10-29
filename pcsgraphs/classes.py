import scipy.sparse as ss
import Lib.copy as clc
import pickle as pkl
import numpy as np
import matplotlib.pyplot as mpl
import math


class DirGraphNode:
    def __init__(self, node_id=None, **labels):
        """inizializzazione di un oggetto DirGraphNode"""
        self.id = node_id
        self.labels = labels
        self.neighbours_out = []   # inizialmente i vicini sono vuoti
        self.neighbours_in = []

    def get_neighbours(self):
        """restituisce una tupla costituita da 2 liste, la prima contenente i neighbours_out e la seconda contenente
        i neighbours_in del nodo"""
        neigh_out = []
        for neigh in self.neighbours_out:  # inserisco in neigh_out tutti gli oggetti DirGraphNode in neighbours_out
            neigh_out.append(neigh[0])  # (andando quindi a prendere solo i nodi e non le etichette relative ai lati)
        return neigh_out, self.neighbours_in

    def degrees(self):
        """restituisce una tupla contenente il grado del nodo rispetto ai lati in uscita e il grado del nodo
        rispetto ai lati in entrata."""
        all_deg = (self.neighbours_out.__len__(), self.neighbours_in.__len__())
        return all_deg

    def get_edge_labels(self, nodes):
        """restituisce la lista di dizionari dei lati che vanno da self ai lati dati in input"""
        edges = []
        for neigh in self.neighbours_out:
            if neigh[0] in nodes:  # l'elemento 0 di neigh è un oggetto DirGraphNode
                edges.append(neigh[1])  # neigh[1] è il dizionario associato al lato con estremità neigh[0]
        return edges  # vado ad aggiungere il dizionario alla lista edges

    def add_neighbours_out(self, nodes_list, **edge_labels):
        """aggiunge nuovi vicini in uscita, con etichette comuni"""
        neighbours_out, _ = self.get_neighbours()
        for node in nodes_list:
            if node in neighbours_out:
                ind_node = neighbours_out.index(node)  # se il nodo esiste già, salvo l'indice per aggiornare
                self.neighbours_out[ind_node][1].update(edge_labels.copy())    # le etichette
            else:
                self.neighbours_out.append((node, edge_labels.copy()))  # se non esiste, aggiungo il nodo con le sue
        return                                                          # etichette

    def add_neighbours_in(self, nodes):
        """aggiunge nuovi vicini in entrata"""
        for node in nodes:
            if node not in self.neighbours_in:   # inserisco il nuovo nodo solo se non è già presente
                self.neighbours_in.append(node)

    def rmv_neighbours_out(self, nodesr):
        """rimuove i vicini in uscita"""
        for node in nodesr:
            for neigh in self.neighbours_out:
                if node in neigh:      # se la chiave node è presente in un elemento di neighbours_out, lo rimuovo
                    self.neighbours_out.remove(neigh)

    def rmv_neighbours_in(self, nodesr):
        """rimuove i vicini in entrata"""
        for node in nodesr:
            if node in self.neighbours_in:
                self.neighbours_in.remove(node)


class DirectedGraph:
    def __init__(self, name='noname_graph', default_weight=1.0, nodesid=None, edgesid=None, node_labels=None,
                 edge_labels=None):
        """inizializzazione di un oggetto DirectedGraph"""
        if nodesid is None:    # questo serve a inizializzare gli attributi con una lista vuota
            nodesid = []       # come operazione di default (questo passaggio è necessario perché gli
        if edgesid is None:    # argomenti di default non possono essere modificabili
            edgesid = []
        if node_labels is None:
            node_labels = {}
        if edge_labels is None:
            edge_labels = {}
        self.name = name
        self.default_weight = default_weight  # il peso di default per i nuovi archi che saranno aggiunti
        self.nodes = {}
        self.add_nodes(nodesid, **node_labels)
        self.add_edges(edgesid, **edge_labels)

    def add_nodes(self, ids, **node_labels):
        """aggiunge al grafo un elenco di nuovi nodi con dizionari comuni"""
        for ID in ids:
            if ID in self.nodes:
                self.nodes.get(ID).labels.update(node_labels.copy())  # se il nodo esiste già, si aggiornano
            else:                                                     # le etichette
                new_node = DirGraphNode(ID, **node_labels.copy())  # altrimenti crea un nuovo nodo e aggiorna
                self.nodes.update({ID: new_node})                  # la lista

    def auto_add_nodes(self, n, **node_labels):
        """aggiunge n nodi al grafo assegnando in automatico un id per ciascuno"""
        ids = []
        i = 0
        while ids.__len__() < n:  # prima creo la lista di id senza generare conflitti
            if i not in self.nodes:
                ids.append(i)
            i += 1
        self.add_nodes(ids, **node_labels)  # poi evoco semplicemente add_nodes con la lista di id creata

    def add_edges(self, edge_list, **labels):
        """aggiunge al grafo nuovi lati con etichette comuni"""
        if 'weight' not in labels.keys():   # se il peso non è presente, aggiungo il valore di default
            labels.update({'weight': self.default_weight})
        for edge in edge_list:
            if edge[0] not in self.nodes.keys():  # se i nodi non esistono, devono essere prima aggiunti al grafo
                self.add_nodes(edge[0])  # aggiungo i nuovi nodi, senza etichette
            if edge[1] not in self.nodes.keys():
                self.add_nodes(edge[1])
        for edge in edge_list:
            self.nodes[edge[0]].add_neighbours_out(self.nodes[edge[1]], **labels.copy())
            self.nodes[edge[1]].add_neighbours_in(self.nodes[edge[0]])
        return

    def rmv_nodes(self, nodes_id_list):
        """rimuove dal grafo una lista di nodi e i lati ad essi connessi"""
        edges_rmv = []
        for i in nodes_id_list:
            for neigh in self.nodes[i].get_neighbours[0]:  # vicini uscenti dal nodo
                edges_rmv.append((i, neigh.node_id))
            for neigh in self.nodes[i].get_neighbours[1]:  # vicini entranti nel nodo
                edges_rmv.append((i, neigh.node_id))
            self.rmv_edges(edges_rmv)  # rimuovo tutti i lati connessi a tutti i nodi
            del self.nodes[i]  # elimino il nodo dalla lista di nodi
        return

    def rmv_edges(self, edges_list):
        """rimuove dal grafo un elenco di lati"""
        for edge in edges_list:
            self.nodes[edge[0]].rmv_neighbours_out(self.nodes[edge[1]])  # rimuovo il vicino uscente dal primo nodo
            self.nodes[edge[1]].rmv_neighbours_in(self.nodes[edge[0]])  # rimuovo il vicino entrante dal secondo nodo
        return

    def get_edges(self):
        """restituisce tutti i lati del grafo"""
        edges = []
        for node in self.nodes:  # ciclo su tutti i nodi
            for neigh in node[1].neighbours_out:  # inserisco nell'elenco di lati tutti lati uscnti da ogni nodo
                edges.append((node[0], neigh[0].id))
        return edges

    def get_edge_labels(self, edges):
        """restituisce una lista di dizionari dei lati in input"""
        labels = []
        for edge in edges:
            if edge in self.get_edges():
                for neigh in self.nodes[edge[0]].neighbours_out:  # self.nodes[edge[0]] è il nodo da cui parte il lato
                    if edge[1] in neigh:         # nei neighbours_out di quel nodo, prendo il neigh che
                        labels.append(neigh[1])  # contiene l'altra estremità del lato e aggiungo a labels
            else:                                # le etichette del lato
                labels.append(None)  # se il lato non esiste, aggiungo None
        return labels

    def get_edge_weight(self, edge):
        """restituisce il peso del lato dato in input"""
        return self.get_edge_labels(edge)[0]['weight']

    def size(self):
        """restituisce il numero di nodi e il numero di lati del grafo"""
        n_edges = self.get_edges().__len__()
        n_nodes = self.nodes.__len__()
        return n_nodes, n_edges

    def copy(self):
        """restituisce una copia del grafo"""
        return clc.deepcopy(self)

    def compute_adjacency(self):
        """restiuisce la matrice di adiacenza del grafo (dopo aver specificato sa calcolarla come densa o sparsa)"""
        matrix = input("digita la tipologia di matrice che desideri (densa o sparsa)")
        mat = []
        if matrix == "densa":
            for node in self.nodes.values():   # prendo il primo nodo e guardo a quali nodi è collegato
                line = []                      # (ovvero formo la prima riga della matrice)
                for node1 in self.nodes.values():  # ciclo nidificato su tutti i nodi, per ogni nodo
                    if node1 in node.get_neighbours()[0]:  # se node1 appartiene alla lista dei vicini uscenti di nodes
                        line.append(self.get_edge_weight([(node.id, node1.id)]))  # aggiungo l'etichetta
                    else:                                                         # relativa al peso alla matrice
                        line.append(0)  # altrimenti aggiungo 0
                mat.append(line)  # infine aggiungo la linea alla matrice
            m = np.array(mat)   # creo una matrice in numpy
            return m
        if matrix == "sparsa":
            m = ss.dok_matrix((self.size()[0], self.size()[0]), dtype=int)  # crea la matrice sparsa
            list_nodes = []
            for node in self.nodes.values():
                list_nodes.append(node)   # creo una lista di oggetti DirGraphNode
            for i in range(self.size()[0]):
                node1 = list_nodes[i]
                for j in range(self.size()[0]):
                    node2 = list_nodes[j]
                    if (node1, node2) in self.get_edges():  # se il lato esiste, aggiungo il suo peso alla matrice
                        mat[i][j] = self.get_edge_labels([(node1, node2)])[0]['weight']
            return m

    def add_from_adjacency(self, adj_mat):
        """data una matrice di adiacenza, aggiunge al grafo nodi e lati caratterizati da essa"""
        self.auto_add_nodes(adj_mat.__len__())  # aggiungo automaticamente i nodi al grafo
        for i in range(adj_mat.__len__()):   # scandisco la matrice e aggiungo i lati
            for j in range(adj_mat.__len__()):
                if adj_mat[i][j] != 0:
                    self.add_edges([(self.nodes[i], self.nodes[j])], **{"weight": adj_mat[i][j]})
                    # si suppone che gli id corrispondano alle posizioni dei nodi nella matrice

    def add_graph(self, new_g):
        """dato un altro grafo, lo si aggiunge all'oggetto grafo corrente"""
        self.nodes.update(new_g.nodes)  # mi è sufficiente copiare tutti i nodi, dato che i lati sono
        return                          # memorizzati all'interno di essi

    def save(self, path="Users/Utente/PycharmProjects/PCS"):
        """salva le proprietà caratterizzanti il grafo in un percorso file"""
        new_folder = self.name
        labels_dict = {}
        with open(path + new_folder + "/id_list.pkl", "wb") as f1:
            pkl.dump(self.nodes.keys, f1)
        with open(path + new_folder + "/adjacency.npz", "wb") as f2:
            ss.save_npz(f2, self.compute_adjacency)  # salvo la matrice di adiacenza
        for node in self.nodes:
            labels_dict.update({node[0]: node[1].labels})  # creo il dizionario id: etichette
        dic = {'name': self.name, 'default_weight': self.default_weight, 'node labels': labels_dict}
        # creo un dizionario con questa struttura: nome, peso di default, dizionario etichette
        with open(path + new_folder + "/attributes.pkl", "wb") as f3:
            pkl.dump(dic, f3)
        edges = self.get_edges()
        labels = self.get_edge_labels(self.get_edges())
        for label in labels:  # elimino il peso dalla lista delle etichette di ogni lato
            label.pop('weight')
        edge_l = dict(zip(edges, labels))  # dizionario delle etichette dei lati
        with open(path + new_folder + "/edge_labels.pkl.", "wb") as f4:
            pkl.dump(edge_l, f4)
        return

    def add_from_files(self, path):
        """dato un percorso file, si aggiunge il grafo caratterizzato dai dati in esso al grafo corrente"""
        with open(path + "/id_list.pkl", "rb") as f1:
            id_list = pkl.load(f1)
        with open(path + "/attributes.pkl", "rb") as f2:  # si suppone che il peso si già inserito negli attributi
            attributes = pkl.load(f2)
        with open(path + "/edge_labels.pkl", "rb") as f3:
            edge_labels = pkl.load(f3)
        graph1 = DirectedGraph(attributes['name'], attributes['default_weight'])  # creo il grafo da aggiungere
        for i in id_list:
            graph1.add_nodes(i, **attributes['node_labels'][i])  # inserisco il nodo con gli attributi relativi
        for i in edge_labels.keys():                             # a esso
            graph1.add_edges(i, **edge_labels[i])
        self.add_graph(graph1)   # uso questo metodo per aggiungere il grafo

    def minpath_dijkstra(self, id1, id2):
        """forniti gli id di 2 nodi, calcola e restituisce il cammino minimo tra essi, con i relativi
        pesi per ogni arco attraversato utilizzando l'algoritmo di dijkstra"""
        for edge in self.get_edges():
            if self.get_edge_weight(edge) < 0:  # se esistono archi con pesi negativi, si restituisce None
                return None, None
        s_node = self.nodes[id1]  # nodo relativo al primo id (inizio)
        f_node = self.nodes[id2]  # nodo relativo al secondo id (fine)
        max_d = 10 ** 8  # valore rappresentativo di infinito
        d = {}  # dizionario delle distanze "attuali" degli altri nodi da id1
        p = {}  # dizionario dei predecessori dei nodi nel cammino minimo "attuale"
        for node in self.nodes:
            d[node] = max_d  # inizialmente, tutti i nodi sono a distanza infinita da id1
            p[node] = None  # nessun nodo ha un predecessore nel cammino minimo
        d[s_node] = 0  # il nodo iniziale ha distanza 0 da se stesso
        q = {}     # q è un dizionario del tipo "nodo: distanza da id1"
        for node in self.nodes:
            q.update({node[0]: max_d})   # inizialmente le distanze sono tutte uguali a infinito
        e = s_node
        b = s_node
        m = max_d
        while q != {}:
            q.pop(e)   # ogni volta che si visita un nodo (quello con distanza "attuale" minore),
            for neigh in e.neighbours_out:                               # esso viene eliminato da q
                u = neigh[0]          # ciclo sui vicini di e
                if d[u] > d[e] + neigh[1]['weight']:  # se la distanza nella tabella è maggiore della distanza del nodo
                    d[u] = d[e] + neigh[1]['weight']  # precende + il peso dell'arco successivo, allora modifico la
                    p[u] = e                          # distanza nella tabella
                    q[u] = d[u]                  # anche il peso in q viene aggiornato
                if neigh[1]['weight'] < m:     # qua memorizzo il vicino di e con arco di peso minore
                    m = neigh[1]['weight']
                    b = neigh[0]
            e = b  # alla fine del ciclo avrò memorizzato quale è l'elemento con distanza minima dal precedente
        b = f_node  # b è inizialmente l'ultimo nodo del percorso
        a = p[f_node]  # parto dall'ultimo nodo e vado indietro passando per il precedente
        path = [f_node]  # aggiungo il primo vertice (ovvero l'ultimo alla lista)
        weights = []
        while p[a] is not None:
            weights.insert(1, self.get_edge_weight((a.id, b.id)))  # aggiungo il peso dell'arco alla lista dei pesi
            path.insert(1, a)  # inserisco il nodo precedente in prima posizione
            b = a  # eseguo queste operazione per passare all'iterazione succesiva
            a = p[a]
        weights = tuple(weights)
        path = tuple(path)
        return path, weights

    def plot(self):
        """permette di visualizzare il grafo (i nodi vengono disposti lungo una circonferenza)"""
        m = self.nodes.__len__()  # numero di nodi del grafo
        n = 0
        coordinates = {}  # dizionario del tipo "nodo.id: coordinate"
        for i in self.nodes.keys():
            if n == 0:
                x = math.cos(0)
                y = math.sin(0)
            else:
                x = math.cos((2 * math.pi)*n / m)  # dispongo i nodi in maniera equidistaziata su una circonferenza
                y = math.sin((2 * math.pi)*n / m)
            n += 1
            coordinates[i] = (x, y)
            mpl.plot(x, y, 'ro', linewidth=2)   # i punti sono rappresentati con un cerchio rosso di dimensione 2
        for (a, b) in self.get_edges():
            xs = [coordinates[a][0], coordinates[b][0]]
            ys = [coordinates[a][1], coordinates[b][1]]
            mpl.plot(xs, ys, 'b-', linewidth=1)    # i lati sono delle linee continue blu di spessore 1
