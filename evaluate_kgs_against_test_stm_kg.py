import evaluate
from kg_builder import *
import numpy as np


def evaluate_kgs(gold_kg_entities: List[Entity], predicted_kg_entities: List[Entity]):
    def entity_to_cluster(e: Entity):
        cluster = set()
        for m in e.mentions:
            cluster.add((m.filename, m.start, m.end, m.text, e.name))
        return frozenset(cluster)
    gold_clusters = [entity_to_cluster(e) for e in gold_kg_entities]
    predicted_clusters = [entity_to_cluster(e) for e in predicted_kg_entities]
    kg = evaluate.Document(predicted_clusters, gold_clusters)
    metric_to_names = {evaluate.muc: 'MUC ',  evaluate.b_cubed: 'b³  ', evaluate.ceafe: 'CEAF'}
    evals = [evaluate.Evaluator(metric) for metric in metric_to_names]
    precisions = []
    recalls  = []
    f1s = []
    print(f'metric\t'
          f'P\t'
          f'R\t'
          f'F1')
    for ev in evals:
        ev.update(kg)
        precisions.append(ev.get_precision())
        recalls.append(ev.get_recall())
        f1s.append(ev.get_f1())
        print(f'{metric_to_names[ev.metric]}\t'
              f'{ev.get_precision() * 100:.1f}\t'
              f'{ev.get_recall() * 100:.1f}\t'
              f'{ev.get_f1() * 100:.1f}')

    print(f'CoNLL\t'
          f'{np.mean(precisions) * 100:.1f}\t'
          f'{np.mean(recalls) * 100:.1f}\t'
          f'{np.mean(f1s) * 100:.1f}')
    tp = set(gold_clusters).intersection(set(predicted_clusters))
    fp = set(predicted_clusters).difference(set(gold_clusters))
    fn = set(gold_clusters).difference(set(predicted_clusters))
    #print('tp', tp)
    #print('fp', [list(c) for c in fp if len(set([f[0] for f in c])) > 1])
    #print('fn', [list(c) for c in fn if len(set([f[0] for f in c])) > 1])

in_domain_stm_test_kg = KGBuilder(collapse_only_within_domains=True, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=True).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs())
cross_domain_stm_test_kg = KGBuilder(collapse_only_within_domains=False, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=True).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs())


stm_in_domain_with_coref = KGBuilder(collapse_only_within_domains=True,  transform_entities_to_singletons=False, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=False).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs())
stm_in_domain_without_coref = KGBuilder(collapse_only_within_domains=True,  transform_entities_to_singletons=True, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=False).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs())
stm_cross_domain_with_coref = KGBuilder(collapse_only_within_domains=False,  transform_entities_to_singletons=False, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=False).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs())
stm_cross_domain_without_coref = KGBuilder(collapse_only_within_domains=False,  transform_entities_to_singletons=True, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=False).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs())

print()
print("in-domain collapsing")
evaluate_kgs(
    in_domain_stm_test_kg.get_kg_entities(),
    stm_in_domain_with_coref.get_kg_entities()
)

print()
print("in-domain collapsing (without coreferences)")
evaluate_kgs(
    in_domain_stm_test_kg.get_kg_entities(),
    stm_in_domain_without_coref.get_kg_entities()
)

print()
print("cross-domain collapsing")
evaluate_kgs(
    cross_domain_stm_test_kg.get_kg_entities(),
    stm_cross_domain_with_coref.get_kg_entities()
)

print()
print("cross-domain collapsing (without coreferences)")
evaluate_kgs(
    cross_domain_stm_test_kg.get_kg_entities(),
    stm_cross_domain_without_coref.get_kg_entities()
)