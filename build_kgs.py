from kg_builder import *

print("Build Test-STM-KG (cross-domain)...")
dump_entities_in_jsonl(
    KGBuilder(collapse_only_within_domains=False, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=True).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs()).get_kg_entities(),
    "knowledge_graph/gold_kg_cross_domain.jsonl"
)

print("Build Test-STM-KG (in-domain)...")
dump_entities_in_jsonl(
    KGBuilder(collapse_only_within_domains=True, filter_uniquely_resolved_entites=True, collapse_by_wikipedia_entities=True).build_kg(STMCorpus("data", without_corefs=False).parse_document_kgs()).get_kg_entities(),
    "knowledge_graph/gold_kg_in_domain.jsonl"
)

print("Build cross-domain KG...")
dump_entities_in_jsonl(
    KGBuilder(collapse_only_within_domains=False).build_kg(SilverLabelledSTMCorpus("data/silver_labelled", without_corefs=False).parse_document_kgs()).get_kg_entities(),
    "knowledge_graph/stm_silver_kg_cross_domain_with_corefs.jsonl"
)

print("Build in-domain KG...")
dump_entities_in_jsonl(
    KGBuilder(collapse_only_within_domains=True).build_kg(SilverLabelledSTMCorpus("data/silver_labelled", without_corefs=False).parse_document_kgs()).get_kg_entities(),
    "knowledge_graph/stm_silver_kg_in_domain_with_corefs.jsonl"
)
