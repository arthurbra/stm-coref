import os
import glob
from collections import Counter
from typing import List, Set
from zipfile import ZipFile
import io
import json
import pandas as pd
import nltk
from phrase_normalizer import normalize_phrase

journal_domain_mapping = {
    'Agricultural and Biological Sciences': 'Agr',
    'Biochemistry, Genetics and Molecular Biology': 'Bio',
    'Chemistry': 'Che',
    'Computer Science': 'CS',
    'Earth and Planetary Sciences': "ES",
    'Engineering': 'Eng',
    'Materials Science': 'MS',
    'Mathematics': 'Mat',
    'Medicine and Dentistry': 'Med',
    'Physics and Astronomy': 'Ast'
}

domain_abbreviations = {
    'Agriculture': 'Agr',
    'Astronomy': 'Ast',
    'Biology': 'Bio',
    'Chemistry': 'Che',
    'Computer_Science': 'CS',
    'Earth_Science': 'ES',
    'Engineering': 'Eng',
    'Materials_Science': 'MS',
    'Mathematics': 'Mat',
    'Medicine': 'Med'
}

# possesives, demonstratives and articles
determiners = set(
    ("my your his her its our their whose "
    + "this that these those "
    + "a an the ").split()
)


class EntityMention:

    def __init__(self, label, start, end, text, domain, filename):
        self.label: str = label
        self.start: int = start
        self.end: int = end
        self.text: str = text
        self.domain: str = domain
        self.filename: str = filename
        # cluster within the file. Is None, if EntityMention does not belong to a Cluster (e.g. Singletons)
        self.cluster: str = None

    @staticmethod
    def from_json(m_json):
        m = EntityMention(label=m_json["label"], start=m_json["start"], end=m_json["end"],
                          text=m_json["text"], domain=m_json["domain"], filename=m_json["filename"])
        m.cluster = m_json.get("cluster")
        return m

    def to_json(self):
        mention_json = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "domain": self.domain,
            "filename": self.filename,
            "cluster": self.cluster,
        }
        return mention_json

    def __str__(self):
        return str(self.label) + " " + str(self.start) + " " + str(self.end) + " "+ self.text

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.label) ^ hash(self.text) ^ hash(self.start) ^ hash(self.end)

    def __eq__(self, other):
        if not isinstance(other, EntityMention):
            return False
        return self.label == other.label and self.start == other.start \
               and self.end == other.end and self.text == other.text

class Cluster:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.entity_mentions = []

    def add_entity(self, entity_mention: EntityMention):
        if self.entity_mentions.count(entity_mention) > 0:
            print("***duplicate coref mention: " + str(entity_mention))
        entity_mention.cluster = self.cluster_id
        self.entity_mentions.append(entity_mention)

    def is_none_cluster(self):
        '''a none cluster contains only non-origin mentions'''
        for e in self.entity_mentions:
            if e.label != "NONE":
                return False
        return True


class Entity:

    def __init__(self, domain:str = None):
        self.mentions: List[EntityMention] = []
        self.domains: Set[str] = set()
        self.name: str = None
        if domain is not None:
            self.domains.add(domain)

    @staticmethod
    def from_json(e_json):
        e = Entity()
        e.domains = set(e_json["domains"])
        e.name = e_json["name"]
        e.mentions = [EntityMention.from_json(m_json) for m_json in e_json["mentions"]]
        return e

    def to_json(self):
        e_json = {
            "name": self.name,
            "domains": list(sorted(self.domains)),
            "top_most_label": self.get_top_most_label(),
            "mentions": [],
        }
        for m in self.mentions:
            e_json["mentions"].append(m.to_json())
        return e_json

    def add_mention(self, mention: EntityMention):
        self.mentions.append(mention)

    def add_mentions(self, mentions: List[EntityMention]):
        self.mentions.extend(mentions)

    def get_labels(self):
        labels = list(sorted(set([m.label for m in self.mentions])))
        if len(labels) > 1 and "NONE" in labels:
            # if an origin mention is coreferent with a non-origin mention (e.g. pronoun), then non-origin mention inherits the label of the origin mention
            labels.remove("NONE")
        return labels

    def get_top_most_label(self):
        c = Counter()
        for m in self.mentions:
            c[m.label] += 1
        if len(c.keys()) > 1 and "NONE" in c.keys():
            del c["NONE"]
        return c.most_common(1)[0][0]

    def is_singleton(self):
        return len(self.mentions) == 1

    def get_origin_mentions(self):
        return [m for m in self.mentions if m.label != "NONE"]

    def is_non_origin_entity(self):
        return len(self.get_origin_mentions()) == 0

    def get_representing_mention(self):
        ''' Returns the mention best representing this entity '''
        representing_mention = None
        for m in self.get_origin_mentions():
            if representing_mention is None or len(m.text) > len(representing_mention.text):
                representing_mention = m
        return representing_mention

    def __str__(self):
        return "(" + ", ".join([m.text for m in self.mentions]) + ")"

    def __repr__(self):
        return self.__str__()

class DocumentKG:
    def __init__(self, text, domain, filename):
        self.text = text
        self.domain = domain
        self.filename = filename
        self.entity_mentions = []
        self.ignored_entity_mentions = []
        self.clusters = []
        self.non_origin_entity_mentions = []

    def add_entity_mention(self, entity_mention: EntityMention):
        self.entity_mentions.append(entity_mention)

    def add_ignored_entity_mention(self, entity_mention: EntityMention):
        print("ignored entity mention in origin corpus:" + str(entity_mention))
        assert self.ignored_entity_mentions.count(entity_mention) == 0
        self.ignored_entity_mentions.append(entity_mention)

    def __get_or_create_cluster(self, cluster_id):
        for c in self.clusters:
            if c.cluster_id == cluster_id:
                return c
        c = Cluster(cluster_id)
        self.clusters.append(c)
        return c

    def __get_entity_mention(self, from_pos, to_pos, text):
        for e in self.entity_mentions:
            if e.start == from_pos and e.end == to_pos and e.text == text:
                return e
        return None

    def __get_cluster_for_mention(self, entity_mention: EntityMention):
        for c in self.clusters:
            for e in c.entity_mentions:
                if e == entity_mention:
                    return c
        return None

    def __is_ignored_mention(self, from_pos, to_pos, text):
        for im in self.ignored_entity_mentions:
            if im.start == from_pos and im.end == to_pos and im.text == text:
                return True
        return False

    def add_mention_to_cluster(self, cluster_id, from_pos, to_pos, text):
        entity_mention = self.__get_entity_mention(from_pos, to_pos, text)
        if entity_mention is None:
            entity_mention = EntityMention("NONE", from_pos, to_pos, text, self.domain, self.filename)
            self.non_origin_entity_mentions.append(entity_mention)

        cluster = self.__get_or_create_cluster(cluster_id)
        cluster.add_entity(entity_mention)

    def get_entity_mention_count(self):
        return len(self.entity_mentions)

    def get_entities(self):
        entities = []
        # singleton entities
        for em in self.entity_mentions:
            c = self.__get_cluster_for_mention(em)
            if c is None:
                e = Entity(self.domain)
                e.add_mention(em)
                entities.append(e)
        for c in self.clusters:
            e = Entity(self.domain)
            e.add_mentions(c.entity_mentions)
            entities.append(e)

        return entities

    def get_entity_count(self):
        count = 0
        # count singleton entities
        for e in self.entity_mentions:
            c = self.__get_cluster_for_mention(e)
            if c is None:
                count += 1
        # count clusters
        count += self.get_proper_cluster_count()
        return count

    def get_non_origin_mention_count(self):
        return len(self.non_origin_entity_mentions)

    def get_none_cluster_count(self):
        '''
        Counts clusters that contain only non origin entity mentions
           (=mentions, that have not been annotated in the origin STM corpus).
        '''
        count = 0
        for c in self.clusters:
            if c.is_none_cluster():
                count += 1
        return count


    def get_proper_cluster_count(self):
        return len(self.clusters) - self.get_none_cluster_count()


def parse_entities(text, document_kg: DocumentKG, ignore_entities=[]):
    for entity_line in text.splitlines():
        anno_inst = entity_line.strip().split("\t")
        if len(anno_inst) == 3:
            entity_type = anno_inst[0].strip()
            anno_inst1 = anno_inst[1].split(" ")
            if len(anno_inst1) == 3:
                keytype, start, end = anno_inst1
            else:
                keytype, start, _, end = anno_inst1
            if entity_type.startswith("T"):
                keyphr_ann = anno_inst[2].strip()
                if keytype in ignore_entities:
                    document_kg.add_ignored_entity_mention(EntityMention(keytype, int(start), int(end), keyphr_ann, document_kg.domain, document_kg.filename))
                else:
                    document_kg.add_entity_mention(EntityMention(keytype, int(start), int(end), keyphr_ann, document_kg.domain, document_kg.filename))


def read_non_split_wiki_entities(entity_resolution_file):
    mention_to_entity = dict()
    df = pd.read_csv(entity_resolution_file, sep="\t", keep_default_na=False)
    df = df[df.apply(lambda x: x['Split Terms'] == '' and x['Wiki IDs'] != '' and x['Wiki IDs'] != ',' and x["Wiki IDs"] != "NILL", axis=1)]
    for _, row in df.iterrows():
        wiki_ids = row['Wiki IDs']
        splits = wiki_ids.strip().split(',')
        # normalise splits
        wiki_id = None
        for s in splits:
            s = s.strip()
            if "wikipedia" in s:
                wiki_id = s
                break
        if wiki_id is None:
            continue
        splits = [s.strip() for s in splits if len(s.strip()) != 0]
        wiki_ids = ",".join(splits)
        domain = domain_abbreviations[row['Domain']]
        mention_to_entity[(domain, row['Filename'], row['Entity'])] = wiki_id
    return mention_to_entity


def parse_clusters(text, document_kg: DocumentKG):
    for entity_line in text.splitlines():
        if entity_line.strip().startswith("#"):
            continue
        anno_inst = entity_line.strip().split("\t")
        if len(anno_inst) == 3:
            entity_type = anno_inst[0].strip()
            anno_inst1 = anno_inst[1].split(" ")
            if len(anno_inst1) == 3:
                keytype, start, end = anno_inst1
            else:
                keytype, start, _, end = anno_inst1
            if entity_type.startswith("T"):
                keyphr_ann = anno_inst[2].strip()
                if keytype.startswith("Cluster"):
                    document_kg.add_mention_to_cluster(keytype, int(start), int(end), keyphr_ann)


class STMCorpus:
    '''Represents the labelled STM-Corpus'''
    def __init__(self, root_folder, without_corefs=False):
        self.root_folder = root_folder
        self.without_corefs = without_corefs

    def create_document_kg(self, domain, filename) -> DocumentKG:
        with open(os.path.join(f"{self.root_folder}/stm-entities", domain,
                               filename.replace(".ann", ".txt")), "r", encoding="utf-8") as f:
            doc_text = f.read()

        document_kg = DocumentKG(doc_text, domain_abbreviations[domain], filename)

        with open(os.path.join(f"{self.root_folder}/stm-entities", domain, filename), "r", encoding="utf-8") as f:
            entities_text = f.read()
            parse_entities(entities_text, document_kg, ["Task", "Result", "Object"])

        if not self.without_corefs:
            with open(os.path.join(f"{self.root_folder}/stm-coref", domain, filename), "r", encoding="utf-8") as f:
                cluster_text = f.read()
                parse_clusters(cluster_text, document_kg)

        return document_kg

    def parse_document_kgs(self) -> List[DocumentKG]:
        result = []
        for f in glob.glob(f"{self.root_folder}/stm-entities/**/*.ann"):
            domain = os.path.basename(os.path.dirname(f))
            file = os.path.basename(f)
            document_kg = self.create_document_kg(domain, file)
            result.append(document_kg)
        return result


class SilverLabelledSTMCorpus:
    '''Represents the silver-labelled STM-Corpus with predicted entities and coreferences.'''
    def __init__(self, root_folder, without_corefs):
        self.root_folder = root_folder
        self.without_corefs = without_corefs
        self.doc_to_domain = dict()
        self.read_domain_mapping()

    def map_to_domain(self, journal_domains):
        for jd in journal_domains:
            if journal_domain_mapping.get(jd) is not None:
                return journal_domain_mapping.get(jd)

    def read_domain_mapping(self):
        with open(f'{self.root_folder}/ccby-domain-mapping.csv', mode="r", encoding="utf-8") as f:
            for l in f:
                id, _, domains = l.split("\t")
                journal_domains = [d.strip() for d in domains.split("|")]
                domain = self.map_to_domain(journal_domains)
                self.doc_to_domain[id + ".ann"] = domain

    def __list_ann_files(self, zip_file):
        filenames = dict()
        with ZipFile(zip_file) as zip:
            for f in zip.namelist():
                if f.endswith(".ann"):
                    name = os.path.basename(f)
                    filenames[name] = f
        return filenames

    def parse_document_kgs(self) -> List[DocumentKG]:
        coref_file_names = self.__list_ann_files(f"{self.root_folder}/coref_predictions_standoff.zip")
        entities_file_names = self.__list_ann_files(f"{self.root_folder}/entities_predictions_standoff.zip")
        print("reading documents...")
        count = 0
        # iterate through intersection of coref and entity predictions
        with ZipFile(f"{self.root_folder}/coref_predictions_standoff.zip") as coref_zip:
            with ZipFile(f"{self.root_folder}/entities_predictions_standoff.zip") as entities_zip:
                for f in coref_file_names.keys() & entities_file_names.keys():
                    name = os.path.basename(f)
                    domain = self.doc_to_domain[name]
                    if domain is None:
                        continue

                    with io.TextIOWrapper(coref_zip.open(
                            coref_file_names[f].replace(".ann", ".txt")), encoding="utf-8") as file:
                        doc_text = file.read()

                    with io.TextIOWrapper(coref_zip.open(coref_file_names[f]), encoding="utf-8") as file:
                        cluster_text = file.read()
                    with io.TextIOWrapper(entities_zip.open(entities_file_names[f]), encoding="utf-8") as file:
                        entities_text = file.read()


                    document_kg = DocumentKG(doc_text, domain, f)
                    parse_entities(entities_text, document_kg, ["Task", "Result", "Object"])
                    if not self.without_corefs:
                        parse_clusters(cluster_text, document_kg)
                    count += 1
                    yield document_kg
                    if count % 100 == 0:
                        print("parsed documents: " + str(count))

        print("parsed documents: " + str(count))


def print_entity_stats(entities, stats_file):
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("domain,label,num_entities,num_singletons,num_clusters,num_mentions,num_coreferent_mentions\n")
        for e in entities:
            if len(e.domains) > 1:
                entity_domain = "MIXED"
            else:
                entity_domain = list(e.domains)[0]

            if len(e.get_labels()) > 1:
                entity_label = "MIXED"
            else:
                entity_label = e.get_labels()[0]
            num_entities = 1
            num_singletons = 0
            num_clusters = 0
            if e.is_singleton():
                num_singletons = 1
            else:
                num_clusters = 1
            f.write(f"{entity_domain},{entity_label},{num_entities},{num_singletons},{num_clusters},0,0\n")

            for m in e.mentions:
                if m.label == "NONE":
                    num_origin_mentions = 0
                else:
                    num_origin_mentions = 1
                if e.is_singleton():
                    num_coreferent_mentions = 0
                else:
                    num_coreferent_mentions = 1
                f.write(f"{m.domain},{m.label},0,0,0,{num_origin_mentions},{num_coreferent_mentions}\n")


def print_kg_entity_stats(entities, stats_file):
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("domain,label,top_most_label, num_mentions,num_entities\n")
        for e in entities:
            if e.is_non_origin_entity():
                continue
            if len(e.domains) > 1:
                domain = "MIXED"
            else:
                domain = list(e.domains)[0]
            if len(e.get_labels()) > 1:
                entity_label = "MIXED"
            else:
                entity_label = e.get_labels()[0]
            f.write(f"{domain},{entity_label},{e.get_top_most_label()},0,1\n")
            for m in e.get_origin_mentions():
                f.write(f"{m.domain},{m.label},{e.get_top_most_label()},1,0\n")


def print_kg_domain_stats(kgs):
    counter = Counter()
    for kg in kgs:
        counter[kg.domain] += 1
    print(counter)


class KGBuilder:
    def __init__(self, collapse_only_within_domains, simple_phrase_normalisation=False, filter_uniquely_resolved_entites=False,
                 collapse_by_wikipedia_entities=False,
                 transform_entities_to_singletons=False):
        self.mention_to_entity_idx = dict()
        self.collapse_only_within_domains = collapse_only_within_domains
        self.simple_phrase_normalisation = simple_phrase_normalisation
        self.filter_uniquely_resolved_entites = filter_uniquely_resolved_entites
        self.collapse_by_wikipedia_entities = collapse_by_wikipedia_entities
        self.transform_entities_to_singletons = transform_entities_to_singletons
        if self.filter_uniquely_resolved_entites:
            # (Domain, Filename, Entity) -> wiki_ids
            self.mention_to_resolved_entity = read_non_split_wiki_entities("knowledge_graph/entity_resolution_annotations.tsv")

    def normalize_mention_text(self, dkg: DocumentKG, m: EntityMention):
        #pos_tags = nltk.pos_tag(m.text.lower().split(), lang="eng", tagset='universal')
        #normalized_tags = [pos[0] for pos in pos_tags if pos[1] != "DET"]

        if not self.simple_phrase_normalisation:
            doc_entity_mentions = [(m.start, m.end) for m in dkg.entity_mentions]
            normalized_text = normalize_phrase(m.start, m.end, dkg.text, doc_entity_mentions)
            return normalized_text
        else:
            tokens = m.text.lower().split()
            normalized_tokens = [t for t in tokens if t not in determiners]

            normalized_text = " ".join(normalized_tokens)
            return normalized_text

    def find_best_kg_entity_or_create(self, domain: str, dkg: DocumentKG, e: Entity):
        normalized_representing_mention_text = self.normalize_mention_text(dkg, e.get_representing_mention())
        if self.collapse_by_wikipedia_entities:
            key = self.__get_unique_wiki_entity(e)
        else:
            key = (domain, normalized_representing_mention_text)

        if key in self.mention_to_entity_idx:
            kg_entity = self.mention_to_entity_idx[key]
        else:
            kg_entity = Entity()
            kg_entity.name = normalized_representing_mention_text
            self.mention_to_entity_idx[key] = kg_entity
        return kg_entity

    def get_kg_entities(self):
        entities = set(self.mention_to_entity_idx.values())
        return entities

    def __get_unique_wiki_entity(self, e: Entity):
        wiki_entities = set()
        for m in e.mentions:
            wiki_entity = self.mention_to_resolved_entity.get((m.domain, m.filename,  m.text))
            if wiki_entity is not None:
                wiki_entities.add(wiki_entity)

        if len(wiki_entities) == 1:
            return list(wiki_entities)[0]
        else:
            if len(wiki_entities) > 1:
                print(f"non unique entities {wiki_entities} for entity {e}:")
            return None

    def _transform_entities_to_singletons(self, entities):
        result = []
        for e in entities:
            for m in e.mentions:
                se = Entity(list(e.domains)[0])
                se.add_mention(m)
                result.append(se)
        return result

    def build_kg(self, document_kgs):
        doc_entity_count = 0
        singlenton_doc_entity_count = 0
        for dkg in document_kgs:
            doc_entities = self._transform_entities_to_singletons(dkg.get_entities()) \
                if self.transform_entities_to_singletons else dkg.get_entities()
            for e in doc_entities:
                if e.is_non_origin_entity():
                    continue

                if self.filter_uniquely_resolved_entites:
                    if self.__get_unique_wiki_entity(e) is None:
                        continue

                if e.is_singleton():
                    singlenton_doc_entity_count += 1
                doc_entity_count += 1
                domain = None
                if self.collapse_only_within_domains:
                    domain = dkg.domain
                kg_entity = self.find_best_kg_entity_or_create(domain, dkg, e)
                kg_entity.domains.add(dkg.domain)
                kg_entity.add_mentions(e.get_origin_mentions())
        print("doc entity count: " + str(doc_entity_count))
        print("doc singleton entity count: " + str(singlenton_doc_entity_count))

        #self.print_entities()
        print("number of KG entities:" + str(len(self.get_kg_entities())))
        return self

    def print_entities(self):
        for key, e in self.mention_to_entity_idx.items():
            mentions = list(sorted(set([e.text for e in e.get_origin_mentions()])))
            normalized_name = key[1]
            if len(mentions) > 1:
                print(f'{normalized_name}: {str(mentions)}')


def kgs_to_entities(kgs):
    return [e for kg in kgs for e in kg.get_entities()]


def print_entities(entities: List[Entity], stats_file, without_mentions=False):
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("name\ttop_most_label\tlabels\tdomains\tnum_mentions\tnum_docs\tnum_domains\tmentions\n")
        for e in entities:
            domains = ";".join(sorted(e.domains))
            labels = ";".join(e.get_labels())
            top_most_label = e.get_top_most_label()
            mentions = [f'{m.text} ({m.label}|{m.domain}|{m.filename})' for m in e.mentions]
            mentions_txt = "\t".join(mentions)
            if without_mentions:
                mentions_txt = ""
            num_mentions = len(e.mentions)
            num_docs = len(set([m.filename for m in e.mentions]))
            num_domains = len(e.domains)
            if num_docs > 1:
                f.write(f"{e.name}\t{top_most_label}\t{labels}\t{domains}\t{num_mentions}\t{num_docs}\t{num_domains}\t{mentions_txt}\n")


def dump_entities_in_jsonl(entities: List[Entity], file):
    with open(file, "w", encoding="utf-8") as f:
        for e in entities:
            e_json = e.to_json()
            json.dump(e_json, f)
            f.write("\n")


def read_entites_from_jsonl(file) -> List[Entity]:
    with open(file, "r", encoding="utf-8") as f:
        result = []
        for line in f:
            e_json = json.loads(line)
            e = Entity.from_json(e_json)
            result.append(e)
        return result


