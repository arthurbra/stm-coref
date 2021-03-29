import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import cohen_kappa_score


def calc_prec_rec_f1(actual, predicted):
    tp = len(set(actual).intersection(predicted))
    fp = len(predicted) - tp
    fn = len(actual) - tp
    precision = 0 if tp + fp == 0 else round(tp / (tp + fp), 4)
    recall = 0 if tp + fn == 0 else round(tp / (tp + fn), 4)
    f1 = 0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 4)
    return precision, recall, f1


def kappa(clusters1, clusters2, entities):
    for c in clusters1:
        c.sort(key=lambda x: x[0])
    for c in clusters2:
        c.sort(key=lambda x: x[0])

    doc1_mentions = [m for c in clusters1 for m in c]
    doc2_mentions = [m for c in clusters2 for m in c]
    shared_mentions = set(doc1_mentions).intersection(set(doc2_mentions))
    all_mentions = list(shared_mentions.union(
        [(start, end) for _, (start, end) in entities]))
    all_mentions.sort(key=lambda x: x[0])

    # extract mention-antecedent pairs from the annotated clusters
    # each mention forms such a pair with each preceding mention
    # mention-antecedent pair == coreference link
    rater1_pairs, rater2_pairs = [], []
    for clusters, pairs in ([clusters1, rater1_pairs], [clusters2, rater2_pairs]):
        for cluster in clusters:
            for i in range(len(cluster)):
                for j in range(i):
                    mention, antecedent = cluster[i], cluster[j]
                    pairs.append((mention, antecedent))

    possible_pairs = []
    for i in range(len(all_mentions)):
        for j in range(i):
            mention, antecedent = all_mentions[i], all_mentions[j]
            are_not_overlapping = antecedent[0] < antecedent[1] < mention[0] < mention[1]
            if are_not_overlapping:
                # with the annotation guidelines used, a pair of overlapping mentions cannot be marked as being
                # coreferent with each other (by assigning them the same cluster-number)
                # thus such a pair is not a possible mention-antecedent pair
                possible_pairs.append((mention, antecedent))

    rater1 = []
    rater2 = []
    for pair in possible_pairs:
        rater1.append(1 if pair in rater1_pairs else 0)
        rater2.append(1 if pair in rater2_pairs else 0)
    
    return np.nan_to_num(cohen_kappa_score(rater1, rater2))


class Document:
    def __init__(self, clusters, gold):
        self.clusters = clusters
        self.gold = gold
        self.mention_to_cluster = {m: tuple(c) for c in clusters for m in c}
        self.mention_to_gold = {m: tuple(c) for c in gold for m in c}


 ###############################################################################
 # code above my code
 # code below copied FROM https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
 ###############################################################################


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, document):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(document.clusters, document.gold)
        else:
            pn, pd = self.metric(document.clusters, document.mention_to_gold)
            rn, rd = self.metric(document.gold, document.mention_to_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

