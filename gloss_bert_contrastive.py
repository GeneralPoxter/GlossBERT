"""
Implementation of Contrastive GlossBERT
Based on: https://github.com/HSLCY/GlossBERT
"""

import argparse
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from nltk.corpus import wordnet

from constants import *
from util import read_annotated_chunks, sense_tagged_chunks


def weak_supervision(tagged_sentence, target_idx):
    sentence = []
    for i, (_, _, _, text) in enumerate(tagged_sentence):
        if i == target_idx:
            sentence.append('"')
        sentence.extend(text)
        if i == target_idx:
            sentence.append('"')
    return " ".join(sentence)


def generate_dataset(paths):
    chunks = read_annotated_chunks(paths)
    sense_chunks = sense_tagged_chunks(chunks)

    dataset = {"gloss1": [], "gloss2": [], "context": [], "labels": []}
    for tagged_sentence in sense_chunks:
        for i, (lemma, gt_sn, _, _) in enumerate(tagged_sentence):
            if lemma and gt_sn:
                senses = wordnet.synsets(lemma)
                context = weak_supervision(tagged_sentence, i)

                for sn1 in gt_sn:
                    gloss1 = f"{lemma}: {senses[sn1 - 1].definition()}"
                    for sn2, ss in enumerate(senses, start=1):
                        if sn2 not in gt_sn:
                            gloss2 = f"{lemma}: {ss.definition()}"

                            dataset["gloss1"].append(gloss1)
                            dataset["gloss2"].append(gloss2)
                            dataset["context"].append(context)
                            dataset["labels"].append(0)

                            dataset["gloss1"].append(gloss2)
                            dataset["gloss2"].append(gloss1)
                            dataset["context"].append(context)
                            dataset["labels"].append(1)

    return Dataset.from_dict(dataset)


def generate_test_dataset(paths, sample_k=None, seed=SEED):
    chunks = read_annotated_chunks(paths)
    sense_chunks = sense_tagged_chunks(chunks)

    words = []
    for sent_idx, tagged_sentence in enumerate(sense_chunks):
        for word_pos, (lemma, gt_sn, _, _) in enumerate(tagged_sentence):
            if lemma and gt_sn:
                words.append((sent_idx, word_pos))

    if sample_k:
        random.seed(seed)
        words = random.sample(words, sample_k)

    dataset = {"gloss1": [], "gloss2": [], "context": [], "label": []}
    # valid_triples tracks indices for evaluating triple classification accuracy
    # gt_sn_map tracks indices for evaluating WSD (word) accuracy
    valid_triples, gt_sn_map, word_idx, triple_idx = [], {}, 0, 0
    for sent_idx, word_pos in words:
        tagged_sentence = sense_chunks[sent_idx]
        lemma, gt_sn, _, _ = tagged_sentence[word_pos]
        senses = wordnet.synsets(lemma)
        context = weak_supervision(tagged_sentence, word_pos)
        gt_sn_map[word_idx] = (gt_sn, len(senses), [])
        for sn1, ss1 in enumerate(senses, start=1):
            for sn2, ss2 in enumerate(senses, start=1):
                if sn1 != sn2:
                    gloss1 = f"{lemma}: {ss1.definition()}"
                    gloss2 = f"{lemma}: {ss2.definition()}"
                    dataset["gloss1"].append(gloss1)
                    dataset["gloss2"].append(gloss2)
                    dataset["context"].append(context)

                    if sn1 in gt_sn and sn2 not in gt_sn:
                        dataset["label"].append(0)
                    else:
                        dataset["label"].append(1)

                    # triple is valid when one sense is gt and the other is not
                    if (sn1 in gt_sn) != (sn2 in gt_sn):
                        valid_triples.append(triple_idx)

                    gt_sn_map[word_idx][-1].append(triple_idx)
                    triple_idx += 1
        word_idx += 1

    return Dataset.from_dict(dataset), valid_triples, gt_sn_map


def truncate_gloss_context_triple(context, gloss1, gloss2, max_length):
    while len(context) + len(gloss1) + len(gloss2) > max_length:
        if len(context) > len(gloss1) + len(gloss2):
            context.pop()
        elif len(gloss1) > len(gloss2):
            gloss1.pop()
        else:
            gloss2.pop()


def tokenize_gloss_context_triple(row, tokenizer):
    gloss1_tokens = tokenizer.tokenize(row["gloss1"])
    gloss2_tokens = tokenizer.tokenize(row["gloss2"])
    ctxt_tokens = tokenizer.tokenize(row["context"])
    truncate_gloss_context_triple(
        ctxt_tokens, gloss1_tokens, gloss2_tokens, tokenizer.model_max_length - 4
    )

    tokens = (
        ["[CLS]"]
        + ctxt_tokens
        + ["[SEP]"]
        + gloss1_tokens
        + ["[SEP]"]
        + gloss2_tokens
        + ["[SEP]"]
    )

    row["input_ids"] = tokenizer.convert_tokens_to_ids(tokens)
    row["attention_mask"] = [1] * len(tokens)
    row["token_type_ids"] = (
        [0] * (len(ctxt_tokens) + 2)
        + [1] * (len(gloss1_tokens) + 1)
        + [1] * (len(gloss2_tokens) + 1)
    )

    padding = [0] * (tokenizer.model_max_length - len(tokens))
    row["input_ids"] += padding
    row["attention_mask"] += padding
    row["token_type_ids"] += padding

    return row


def compute_metrics(eval_pred):
    probabilities, labels = eval_pred
    pred_labels = probabilities.argmax(axis=1)
    acc = sum(pred_labels == labels) / len(labels)
    return {"accuracy": acc}


def compute_metrics_test(eval_pred, valid_triples, gt_sn_map):
    probabilities, labels = eval_pred
    pred_labels = probabilities.argmax(axis=1)
    triple_acc = (pred_labels[valid_triples] == labels[valid_triples]).mean()

    # Perform WSD inference
    word_correct, word_total = 0, 0
    for gt_sn, n, idxs in gt_sn_map.values():
        if len(idxs) and idxs[-1] >= len(pred_labels):
            continue

        scores = [0] * n
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    if pred_labels[idxs[k]] == 0:
                        scores[i] += 1
                    else:
                        scores[j] += 1
                    k += 1

        pred_sn = scores.index(max(scores)) + 1
        word_correct += int(pred_sn in gt_sn)
        word_total += 1

    word_acc = word_correct / word_total

    return {
        "triple_accuracy": triple_acc,
        "word_accuracy": word_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--model-name", type=str, default="glossBERT-contrastive")
    args = parser.parse_args()

    if args.mode == "train":
        glossBERT = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        train = generate_dataset(TRAIN_PATHS)
        eval = generate_dataset(DEV_PATHS)

        train_data = (
            train.shuffle(seed=SEED)
            .select(range(200000))
            .map(tokenize_gloss_context_triple, fn_kwargs={"tokenizer": tokenizer})
        )
        eval_data = (
            eval.shuffle(seed=SEED)
            .select(range(5000))
            .map(tokenize_gloss_context_triple, fn_kwargs={"tokenizer": tokenizer})
        )

        training_args = TrainingArguments(
            eval_strategy="epoch",
            logging_strategy="epoch",
            output_dir=f"{args.model_name}-training",
            save_steps=5000,
            learning_rate=1e-5,
            warmup_ratio=0.1,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_on_start=True,
            full_determinism=True,
            report_to="none",
        )

        trainer = Trainer(
            model=glossBERT,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(args.model_name)

    elif args.mode == "eval":
        glossBERT = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        test, valid_triples, gt_sn_map = generate_test_dataset(
            TEST_PATHS, sample_k=TEST_SAMPLE
        )
        test_data = test.map(
            tokenize_gloss_context_triple, fn_kwargs={"tokenizer": tokenizer}
        )

        evaluator_args = TrainingArguments(
            output_dir=f"{args.model_name}-evaluation",
            per_device_eval_batch_size=16,
            full_determinism=True,
            report_to="none",
        )

        evaluator = Trainer(
            model=glossBERT,
            args=evaluator_args,
            eval_dataset=test_data,
            compute_metrics=lambda eval_pred: compute_metrics_test(
                eval_pred, valid_triples, gt_sn_map
            ),
        )

        eval_results = evaluator.evaluate()
        print(eval_results)

    else:
        raise Exception("Unknown mode")
