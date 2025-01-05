"""
Simplified implementation of GlossBERT
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

    dataset = {"gloss": [], "context": [], "labels": []}
    for tagged_sentence in sense_chunks:
        for i, (lemma, gt_sn, _, _) in enumerate(tagged_sentence):
            if lemma and gt_sn:
                for sn, ss in enumerate(wordnet.synsets(lemma), start=1):
                    gloss = f"{lemma}: {ss.definition()}"
                    context = weak_supervision(tagged_sentence, i)
                    label = int(sn in gt_sn)

                    dataset["gloss"].append(gloss)
                    dataset["context"].append(context)
                    dataset["labels"].append(label)

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

    dataset = {"gloss": [], "context": [], "label": []}
    gt_sn_map, word_idx, pair_idx = {}, 0, 0
    for sent_idx, word_pos in words:
        tagged_sentence = sense_chunks[sent_idx]
        lemma, gt_sn, _, _ = tagged_sentence[word_pos]
        gt_sn_map[word_idx] = (gt_sn, [])
        for sn, ss in enumerate(wordnet.synsets(lemma), start=1):
            gloss = f"{lemma}: {ss.definition()}"
            context = weak_supervision(tagged_sentence, word_pos)
            label = int(sn in gt_sn)
            dataset["gloss"].append(gloss)
            dataset["context"].append(context)
            dataset["label"].append(label)
            gt_sn_map[word_idx][1].append(pair_idx)
            pair_idx += 1
        word_idx += 1

    return Dataset.from_dict(dataset), gt_sn_map


def truncate_gloss_context_pair(context, gloss, max_length):
    while len(context) + len(gloss) > max_length:
        if len(context) > len(gloss):
            context.pop()
        else:
            gloss.pop()


def tokenize_gloss_context_pair(row, tokenizer):
    gloss_tokens = tokenizer.tokenize(row["gloss"])
    ctxt_tokens = tokenizer.tokenize(row["context"])
    truncate_gloss_context_pair(
        ctxt_tokens, gloss_tokens, tokenizer.model_max_length - 3
    )

    tokens = ["[CLS]"] + ctxt_tokens + ["[SEP]"] + gloss_tokens + ["[SEP]"]

    row["input_ids"] = tokenizer.convert_tokens_to_ids(tokens)
    row["attention_mask"] = [1] * len(tokens)
    row["token_type_ids"] = [0] * (len(ctxt_tokens) + 2) + [1] * (len(gloss_tokens) + 1)

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


def compute_metrics_test(eval_pred, gt_sn_map):
    probabilities, labels = eval_pred
    pred_labels = probabilities.argmax(axis=1)
    pair_acc = sum(pred_labels == labels) / len(labels)

    # Perform WSD inference
    word_correct, word_total = 0, 0
    for gt_sn, idxs in gt_sn_map.values():
        if max(idxs) >= len(probabilities):
            continue
        pred_sn = probabilities[idxs, 1].argmax() + 1
        word_correct += int(pred_sn in gt_sn)
        word_total += 1
    word_acc = word_correct / word_total

    return {"pair_accuracy": pair_acc, "word_accuracy": word_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--model-name", type=str, default="glossBERT")
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
            .map(tokenize_gloss_context_pair, fn_kwargs={"tokenizer": tokenizer})
        )
        eval_data = (
            eval.shuffle(seed=SEED)
            .select(range(5000))
            .map(tokenize_gloss_context_pair, fn_kwargs={"tokenizer": tokenizer})
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

        test, gt_sn_map = generate_test_dataset(TEST_PATHS, sample_k=TEST_SAMPLE)
        test_data = test.map(
            tokenize_gloss_context_pair, fn_kwargs={"tokenizer": tokenizer}
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
                eval_pred, gt_sn_map
            ),
        )

        eval_results = evaluator.evaluate()
        print(eval_results)

    else:
        raise Exception("Unknown mode")
