from typing import Callable, Iterable, List, Tuple, Any

import spacy
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import registry
from thinc.api import Model, chain, with_getitem, Linear, Logistic
from thinc.types import Floats2d, Ints1d, Ragged
from thinc.api import reduce_mean


def _ensure_extensions() -> None:
    if not Doc.has_extension("rel"):
        Doc.set_extension("rel", default={})


@Language.factory("relation_extractor")
def make_relation_extractor(
    nlp: Language,
    name: str,
    model: Model,
    threshold: float = 0.5,
) -> "RelationExtractor":
    _ensure_extensions()
    return RelationExtractor(nlp.vocab, model, name=name, threshold=threshold)


class RelationExtractor(TrainablePipe):
    """
    Trainable relation extraction component.

    Stores relation scores in `doc._.rel` with the convention:
      doc._.rel[(ent1.start, ent2.start)] = {label: score, ...}
    """

    def __init__(self, vocab, model: Model, name: str = "relation_extractor", threshold: float = 0.5):
        _ensure_extensions()
        self.vocab = vocab
        self.model = model
        self.name = name
        self.threshold = threshold
        self.cfg = {"labels": []}

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self.cfg.get("labels", []))

    def add_label(self, label: str) -> None:
        if label not in self.cfg["labels"]:
            self.cfg["labels"] = list(self.labels) + [label]

    def initialize(self, get_examples, *, nlp=None, labels: Iterable[str] | None = None):
        if labels:
            for lab in labels:
                self.add_label(lab)
        examples = list(get_examples())
        for eg in examples:
            for _, lab_dict in eg.reference._.rel.items():
                for lab in lab_dict:
                    self.add_label(lab)
            # During RE training, we rely on gold entities. The `Example` provides
            # them on the reference doc, so we copy them to the predicted doc
            # before initializing the model.
            eg.predicted.ents = eg.reference.ents
        if examples:
            self.model.initialize(X=[eg.predicted for eg in examples])
        return self

    def predict(self, docs: List[Doc]) -> Floats2d:
        return self.model.predict(docs)

    def set_annotations(self, docs: List[Doc], scores: Floats2d) -> None:
        get_instances = self.model.attrs.get("get_instances")
        if get_instances is None:
            raise ValueError("Relation model is missing get_instances attr")

        n_labels = len(self.labels)
        if n_labels == 0:
            return

        offset = 0
        for doc in docs:
            instances = get_instances(doc)
            doc._.rel = {}
            for i, (e1, e2) in enumerate(instances):
                key = (e1.start, e2.start)
                row = scores[offset + i]
                doc._.rel[key] = {lab: float(row[j]) for j, lab in enumerate(self.labels)}
            offset += len(instances)

    def update(
        self,
        examples: List[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd=None,
        losses=None,
    ):
        if losses is None:
            losses = {}
        # Use gold entities during training by copying reference ents.
        docs = []
        for eg in examples:
            eg.predicted.ents = eg.reference.ents
            docs.append(eg.predicted)
        scores, backprop = self.model.begin_update(docs)
        loss, d_scores = self.get_loss(examples, scores)
        backprop(d_scores)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] = losses.get(self.name, 0.0) + float(loss)
        if set_annotations:
            self.set_annotations(docs, scores)
        return losses

    def score(self, examples: Iterable[Example], **kwargs):
        """Calculates exact-match micro accuracy for relations."""
        correct = 0
        total = 0
        for eg in examples:
            gold_rel = eg.reference._.rel
            pred_rel = eg.predicted._.rel
            
            get_instances = self.model.attrs.get("get_instances")
            instances = get_instances(eg.reference)
            for e1, e2 in instances:
                key = (e1.start, e2.start)
                
                # Gold
                g_scores = gold_rel.get(key, {})
                g_max = max(g_scores.keys(), key=lambda k: g_scores[k]) if g_scores else None
                
                # Pred
                p_scores = pred_rel.get(key, {})
                p_max = max(p_scores.keys(), key=lambda k: p_scores[k]) if p_scores else None
                
                if g_max is not None:
                    total += 1
                    if p_max == g_max:
                        correct += 1
                        
        acc = correct / max(total, 1)
        return {"relation_extractor": acc}

    def get_loss(self, examples: List[Example], scores: Floats2d):
        """
        Multi-label logistic loss over candidate entity pairs.
        """
        import numpy as np

        get_instances = self.model.attrs.get("get_instances")
        if get_instances is None:
            raise ValueError("Relation model is missing get_instances attr")

        labels = list(self.labels)
        n_labels = len(labels)

        gold = []
        for eg in examples:
            instances = get_instances(eg.reference)
            rel = eg.reference._.rel or {}
            for (e1, e2) in instances:
                key = (e1.start, e2.start)
                lab_scores = rel.get(key, {})
                row = [float(lab_scores.get(lab, 0.0)) for lab in labels]
                gold.append(row)

        if not gold:
            return 0.0, scores * 0.0

        Y = np.asarray(gold, dtype="float32")
        X = scores
        # logistic loss: -(y*log(x) + (1-y)*log(1-x))
        eps = 1e-6
        Xc = np.clip(X, eps, 1 - eps)
        loss = -((Y * np.log(Xc)) + ((1 - Y) * np.log(1 - Xc))).mean()
        d_scores = (X - Y) / (Y.shape[0] * n_labels)
        return loss, d_scores


@registry.architectures("rel_classification_layer.v1")
def create_classification_layer(nO: int = None, nI: int = None) -> Model:
    return chain(Linear(nO=nO, nI=nI), Logistic())


def _instance_init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model


def _instance_forward(model: Model, docs: List[Doc], is_train: bool):
    import numpy as np

    pooling = model.get_ref("pooling")
    tok2vec = model.get_ref("tok2vec")
    get_instances = model.attrs["get_instances"]

    all_instances = [get_instances(doc) for doc in docs]
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)
    width = tokvecs[0].shape[1] if tokvecs else 0

    ents = []
    lengths = []
    for instances, tokvec in zip(all_instances, tokvecs):
        token_indices = []
        for e1, e2 in instances:
            for ent in (e1, e2):
                token_indices.extend(list(range(ent.start, ent.end)))
                lengths.append(ent.end - ent.start)
        if token_indices:
            ents.append(tokvec[token_indices])
        else:
            ents.append(tokvec[0:0])

    if len(lengths) == 0:
        relations = model.ops.alloc2f(0, width * 2)

        def backprop(d_relations: Floats2d) -> List[Doc]:
            return bp_tokvecs([t * 0.0 for t in tokvecs])

        return relations, backprop

    lengths_arr: Ints1d = model.ops.asarray(lengths, dtype="int32")
    entities = Ragged(model.ops.flatten(ents), lengths_arr)
    pooled, bp_pooled = pooling(entities, is_train)
    relations = model.ops.reshape2f(pooled, -1, pooled.shape[1] * 2)

    def backprop(d_relations: Floats2d) -> List[Doc]:
        d_pooled = model.ops.reshape2f(d_relations, d_relations.shape[0] * 2, -1)
        d_ents = bp_pooled(d_pooled).data

        d_tokvecs = []
        ent_index = 0
        for doc_nr, instances in enumerate(all_instances):
            shape = tokvecs[doc_nr].shape
            d_tokvec = model.ops.alloc2f(*shape)
            count_occ = model.ops.alloc2f(*shape)
            for e1, e2 in instances:
                for ent in (e1, e2):
                    d_tokvec[ent.start : ent.end] += d_ents[ent_index]
                    count_occ[ent.start : ent.end] += 1
                    ent_index += ent.end - ent.start
            d_tokvec /= count_occ + 1e-11
            d_tokvecs.append(d_tokvec)
        return bp_tokvecs(d_tokvecs)

    return relations, backprop


@registry.architectures("rel_instance_tensor.v1")
def create_tensors(
    tok2vec: Model,
    pooling: Model,
    get_instances: Callable[[Doc], List[Tuple[Span, Span]]],
) -> Model:
    return Model(
        "instance_tensors",
        _instance_forward,
        layers=[tok2vec, pooling],
        refs={"tok2vec": tok2vec, "pooling": pooling},
        attrs={"get_instances": get_instances},
        init=_instance_init,
    )


@registry.misc("rel_instance_generator.v1")
def create_instances(max_length: int = 100) -> Callable[[Doc], List[Tuple[Span, Span]]]:
    def get_instances(doc: Doc) -> List[Tuple[Span, Span]]:
        instances: list[tuple[Span, Span]] = []
        ents = list(doc.ents)
        for ent1 in ents:
            for ent2 in ents:
                if ent1 is ent2:
                    continue
                if max_length and abs(ent2.start - ent1.start) > max_length:
                    continue
                instances.append((ent1, ent2))
        return instances

    return get_instances


@registry.architectures("rel_model.v1")
def create_relation_model(
    create_instance_tensor: Model,
    classification_layer: Model,
) -> Model:
    model = chain(create_instance_tensor, classification_layer)
    model.attrs["get_instances"] = create_instance_tensor.attrs["get_instances"]
    return model

