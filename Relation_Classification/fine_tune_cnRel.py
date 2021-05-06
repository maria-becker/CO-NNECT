import argparse
import glob
import logging
import os
import random
from sklearn.metrics import classification_report
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import BCEWithLogitsLoss

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)



#try:
from torch.utils.tensorboard import SummaryWriter
#except ImportError:
#    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

LABEL_FILE = 'labels-add-random.txt'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def multi_results(preds, labels, label_map, threshold:dict, global_threshold=True):

    assert len(label_map ) == labels.shape[1]
    len_labels = labels.shape[1]

    tp = np.array([0.0] * len_labels)
    fn = np.array([0.0] * len_labels)
    fp = np.array([0.0] * len_labels)
    tn = np.array([0.0] * len_labels)

    for i in range(len(preds)):
        one_pred = preds[i]
        one_label = labels[i]
        goldanno, predanno = [], []
        for key in label_map.keys():
            label = label_map[key]
            goldanno.append(int(one_label[key]))
            if not global_threshold:
                predanno.append(int(one_pred[key] >= threshold[label]))
            else:
                predanno.append(int(one_pred[key] >= threshold['GLOBAL']))
        goldanno = np.array(goldanno)
        predanno = np.array(predanno)
        tp += (goldanno & predanno)
        fp += (predanno - goldanno > 0)
        fn += (goldanno - predanno > 0)
        tn += ((goldanno == 0) & (predanno == 0))

    prec_class = tp / (tp + fp)
    rec_class = tp / (tp + fn)
    acc_class = (tp + tn) / (tp + fp + fn + tn)
    f1_class = 2 * prec_class * rec_class / (prec_class + rec_class)

    return prec_class, rec_class, acc_class, f1_class

heads_to_prune={0:[0,2,4,6,8,10],1:[0,2,4,6,8,10],2:[0,2,4,6,8,10],3:[0,2,4,6,8,10],4:[0,2,4,6,8,10],5:[0,2,4,6,8,10]}

class RelationData(Dataset):

    def __init__(self, path, label_map, tokenizer, max_len, targets_exit = True):
        self.targets_exit = targets_exit
        self.sources = []
        self.labels =[]
        self.stoi = label_map
        self.itos = {i: l for l, i in label_map.items()}
        sources = [s.strip('\n').split() for s in open(path + ".src", "r").readlines()]
        if targets_exit:
            targets = [int(s.strip("\n")) for s in open(path + ".trg").readlines()]
            for s, i in zip(sources, targets):
                if 2 < len(s) <= max_len:
                    self.sources.append(
                        tokenizer.encode_plus(s, max_length=max_len, pad_to_max_length=True, return_tensors="pt"))
                    self.labels.append(i)
        else:
            for s in sources:
                if 2 < len(s) <= max_len:
                    self.sources.append(
                        tokenizer.encode_plus(s, max_length=max_len, pad_to_max_length=True, return_tensors="pt"))


    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        if self.targets_exit:
            return self.sources[i]["input_ids"].squeeze(), self.sources[i]["attention_mask"].squeeze(), torch.tensor(self.labels[i])
        else:
            return self.sources[i]["input_ids"].squeeze(), self.sources[i]["attention_mask"].squeeze()

class MultiRelationData(Dataset):

    def __init__(self, path, label_map, tokenizer, max_len):
        self.sources = []
        self.labels =[]
        self.stoi = label_map
        self.itos = {i: l for l, i in label_map.items()}
        sources = [s.split() for s in open(path + ".src", "r").read().split('\n')]
        targets = [s.strip("\n") for s in open(path + ".trg").readlines()]
        for s, labels in zip(sources, targets):
            if len(s) <= max_len:
                self.sources.append(tokenizer.encode_plus(s, max_length=max_len, pad_to_max_length=True,return_tensors="pt"))
                label_list = [int(l) for l in labels.split(',') ]
                one_hot_targets = np.zeros(len(label_map))
                for i in label_list:
                    one_hot_targets[i] = 1

                self.labels.append(one_hot_targets)
            # else:
            #    print(s, len(s), i, self.itos[i])


    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return self.sources[i]["input_ids"].squeeze(), self.sources[i]["attention_mask"].squeeze(), torch.tensor(self.labels[i])


def train(args, train_dataset, model, tokenizer, eval_dataset=None,label_map=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    # fine tune specific heads # half
    model.prune_heads(heads_to_prune)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            #print(" before len batch", len(batch), batch)
            batch = tuple(t.to(args.device) for t in batch)
            #print(" after len batch", len(batch), batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            #xent_loss = outputs[0]
            #logits = outputs[1]
            #loss_fct =

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        assert eval_dataset is not None
                        results = evaluate(args, model, eval_dataset,label_map=label_map)
                        #for key, value in results.items():
                            #if "loss" in key:
                        tb_writer.add_scalar("eval_loss", results['eval_loss'], global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def true_predictions( preds, labels, label_map):
    #with open(file,"w") as f:
    predictions = []
    for p, l in zip(preds, labels):
            ps = [(label_map[i] + " " + str(p[i])[:4]) for i in range(len(label_map)) if p[i] >= p[l]]
            predictions.append(ps)
    return predictions


def multiple_predictions(preds, label_map, threshold:dict, global_threshold=True):
    predictions = []
    for p in preds:
        if not global_threshold:
            ps = [(label_map[i] + " " + str(p[i])[:5]) for i in range(len(label_map)) if p[i] >= threshold[label_map[i]]]
        else:
            ps = [(label_map[i] + " " + str(p[i])[:5]) for i in range(len(label_map)) if
                  p[i] >= threshold['GLOBAL']]
        predictions.append(ps)
    return predictions

def max_predictions(preds, p_ids, label_map):
    predictions = []
    for i, p in enumerate(p_ids):
        predictions.append(label_map[p] + " " + str(preds[i][p])[:5])
    return predictions

def all_predictions(preds,  label_map):
    # with open(file,"w") as f:
    predictions = []
    for p in preds:
        ps = [(label_map[i] + " " + str(p[i])[:4]) for i in range(len(label_map)) ]
        predictions.append(ps)
    return predictions

def evaluate_multilabel(args, model, eval_dataset, label_map, threshold:dict, prefix = '', multilabel_data = True, global_threshold=True):
    if global_threshold:
        eval_output_dir = os.path.join("threshold_" + str(threshold['GLOBAL']), args.output_dir)
    else:
        eval_output_dir = args.output_dir

    results = {}
    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label = None
    batch_num = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_num += 1
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            if multilabel_data:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids

            outputs = model(**inputs)
            if inputs['labels'] is not None:
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            else:
                logits = outputs[0]


        nb_eval_steps += 1
        # preds.append(logits.detach().cpu().numpy())
        # out_label_ids.append(inputs["labels"].detach().cpu().numpy())
        # probs = softmax(logits)
        probs = torch.nn.functional.sigmoid(logits)
        if preds is None:
            preds = probs.detach().cpu().numpy()
            out_label = batch[2].detach().cpu().numpy()
        #   print("logits shape", logits.shape)

        #    print("preds shape", preds.shape)

        else:
            #    print("preds shape", preds.shape, "logits shape", logits.shape)
            preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)

            out_label = np.append(out_label, batch[2].detach().cpu().numpy(), axis=0)

        # if args.output_mode == "classification":
    if not multilabel_data:
        out_label_ids = np.eye(len(label_map))[out_label]
    else:
        out_label_ids = out_label

    #print("preds shape", preds.shape)
    #preds_ids = np.argmax(preds, axis=1)
    preds_ids = []
    for pred in preds:
        #print('label map keys', label_map.keys())
        #print('threshold keys', threshold.keys())
        if global_threshold:
            larger_than_threshold = [1 if pred[i] > threshold['GLOBAL'] else 0 for i in range(len(pred))]
        else:
            larger_than_threshold = [1 if pred[i] > threshold[label_map[i]] else 0 for i in range(len(pred)) ]
        preds_ids.append(larger_than_threshold)

    preds_ids = np.array(preds_ids)
    #print('preds ids shape', preds_ids.shape)
    #print('out_label_ids shape', out_label_ids.shape)

    result = simple_accuracy(preds_ids, out_label_ids)
    labels = [i for i, l in label_map.items()]
    target_names = [l for i, l in label_map.items()]
    result_report = classification_report(preds_ids, out_label_ids, labels=labels, target_names=target_names)
    # print("result_report", result_report)
    eval_loss = eval_loss / nb_eval_steps
    if eval_loss == 0.0:
        results['eval_loss'] = "Eval loss is not computed."
    else:
        results["eval_loss"] = eval_loss
    results["accuracy"] = result

    prec_class , rec_class, acc_class, f1_class = multi_results(preds, out_label_ids,label_map,threshold,global_threshold=global_threshold)

    results['accuracy_class'] = acc_class

    results["multi_predictions"] = multiple_predictions(preds, label_map,threshold,global_threshold=global_threshold)
    if prefix == "":
        prefix = args.test_data + "_global_threshold"
    output_eval_file = os.path.join(eval_output_dir, prefix + "_eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            if key != "multi_predictions":
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
        writer.write(result_report)
    return results

def evaluate(args, model, eval_dataset, label_map, all_logits=True, prefix=""):
   # eval_task_names = (args.task_name,)
    #softmax = torch.nn.Softmax(dim=-1)

    eval_output_dir = args.output_dir

    results = {}
    #for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        #eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    batch_num = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch_num += 1
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if len(batch) == 3:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert"] else None
                        )  # XLM and DistilBERT don't use segment_ids
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1

                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    outputs = model(**inputs)
                    logits = outputs[0]


            # preds.append(logits.detach().cpu().numpy())
            # out_label_ids.append(inputs["labels"].detach().cpu().numpy())
            # probs = softmax(logits)
            probs = torch.nn.functional.sigmoid(logits)
            if 'labels' in inputs:
                if preds is None:
                    preds = probs.detach().cpu().numpy()

                    out_label_ids = inputs["labels"].detach().cpu().numpy()
            #   print("logits shape", logits.shape)

            #    print("preds shape", preds.shape)

                else:
                    #      print("preds shape", preds.shape, "logits shape", logits.shape)
                    preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)

                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            else:
                if preds is None:
                    preds = probs.detach().cpu().numpy()
                else:
                    #      print("preds shape", preds.shape, "logits shape", logits.shape)
                    preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)

                out_label_ids = None



            #if args.output_mode == "classification":

    #print("preds shape", preds.shape)
    preds_ids = np.argmax(preds, axis=1)
    if 'labels' in inputs and out_label_ids is not None:
        result = simple_accuracy(preds_ids, out_label_ids)
        labels = [i for i,l in label_map.items()]
        target_names = [l for i,l in label_map.items()]
        result_report = classification_report(preds_ids,out_label_ids, labels=labels, target_names=target_names)
        #print("result_report", result_report)
        if eval_loss != 0.0 and nb_eval_steps != 0:
            eval_loss = eval_loss / nb_eval_steps
            results["eval_loss"] = eval_loss
        results["accuracy"] = result
        output_eval_file = os.path.join(eval_output_dir, prefix + "_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
            writer.write(result_report)


    if not all_logits:
            results["multi_predictions"] = true_predictions(preds,out_label_ids, label_map)
    else:
            prefix += "_all_preds"
            results["multi_predictions"] = all_predictions(preds, label_map)
    if prefix == "":
        prefix = args.test_data

    return results

def get_prediction_multilabel(args, model, eval_dataset, label_map, threshold:dict, prefix = '', global_threshold=True):
    eval_output_dir = args.output_dir
    results = {}
    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    batch_num = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_num += 1
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids

            outputs = model(**inputs)
            #print("outputs", len(outputs), outputs[0])
            logits = outputs[0]
        probs = torch.nn.functional.sigmoid(logits)
        if preds is None:
            preds = probs.detach().cpu().numpy()
        else:
            #    print("preds shape", preds.shape, "logits shape", logits.shape)
            preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)

    preds_ids = np.argmax(preds, axis=1)

    results["max_predictions"] = max_predictions(preds, preds_ids, label_map)
    results["multi_predictions"] = multiple_predictions(preds, label_map,threshold,global_threshold=global_threshold)
    return results

def get_prediction_max(args, model, eval_dataset, label_map,  prefix = ''):
    eval_output_dir = args.output_dir
    results = {}
    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    batch_num = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_num += 1
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids

            outputs = model(**inputs)
            #print("outputs", len(outputs), outputs[0])
            logits = outputs[0]
        probs = torch.nn.functional.sigmoid(logits)
        if preds is None:
            preds = probs.detach().cpu().numpy()
        else:
            #    print("preds shape", preds.shape, "logits shape", logits.shape)
            preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)

    preds_ids = np.argmax(preds, axis=1)

    results["max_predictions"] = max_predictions(preds,preds_ids,label_map)
    return results



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/cnREL",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--train_data",
        default="train",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--test_data",
        default="test",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=15,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument('--do_eval_multilabels', action='store_true', help='To run evaluation of multilabel evaluation')
    parser.add_argument('--do_multiple_prediction', action='store_true', help='To obtain predictions without evaluation')
    parser.add_argument('--do_max_prediction', action='store_true', help='To obtain max predictions without evaluation')
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=3000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare XNLI task
    #args.task_name = "xnli"
    #if args.task_name not in processors:
    #    raise ValueError("Task not found: %s" % (args.task_name))
    #processor = processors[args.task_name](language=args.language, train_language=args.train_language)
    #args.output_mode = output_modes[args.task_name]
    #label_list = processor.get_labels()
    #num_labels = len(label_list)

    label_map = {s.split()[0]: int(s.split()[1].strip("\n")) for s in open(os.path.join(args.data_dir,LABEL_FILE)).readlines()}
    i_to_label = {i:l for l,i in label_map.items()}

    num_labels = len(label_map)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir,
    )
    args.model_type = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_path = os.path.join(args.data_dir, args.train_data)
        train_dataset = RelationData(train_path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer)
        if args.evaluate_during_training:
            path = os.path.join(args.data_dir, args.test_data)
            eval_dataset = RelationData(path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset=eval_dataset,
                                         label_map=i_to_label)
        else:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        test_path = os.path.join(args.data_dir, args.test_data)
        test_dataset = RelationData(test_path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            #prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            prefix = args.test_data
            all_logits = False
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, eval_dataset=test_dataset, label_map=i_to_label, prefix=prefix, all_logits=all_logits)

            if all_logits:
                    prefix += "_all_preds"
            predict_output_file = os.path.join(args.output_dir, str(global_step) + prefix + "_predict_output.txt")
            sources = [s for s in open(test_path + ".src", "r").read().split('\n')]
            targets = [i_to_label[int(s.strip("\n"))] for s in open(test_path + ".trg").readlines()]

            with open(predict_output_file,"w") as f:
                width_s = max([len(s) for s in sources]) + 2
                width_t = max([len(t) for t in targets]) + 2
                headers = ["input_concepts", "gold_label", "predictions"]
                report = headers[0] + " "*(width_s - len(headers[0])) + \
                                           headers[1] + " "*(width_t - len(headers[1])) + headers[2]
                report += '\n\n'
                for s,t,p in zip(sources, targets, result["multi_predictions"]):
                    row = s + " "*(width_s - len(s)) + \
                                           t + " "*(width_t - len(t)) + str(p)
                    report += row
                    report +="\n"
                f.write(report)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())

            results.update(result)

    if args.do_eval_multilabels and args.local_rank in [-1, 0]:

        import csv
        thresholds_dict = dict()

        def to_float(str_num):
            new_str = '.'.join(str_num.split(','))
            return float(new_str)

        with open('thresholds.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                thresholds_dict.update({row[0].strip(): to_float(row[1])})

        multilabel_data = False
        global_threshold = True
        thresholds_dict['GLOBAL'] = 0.97

        test_path = os.path.join(args.data_dir, args.test_data)
        if multilabel_data:
            test_dataset = MultiRelationData(test_path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer)
        else:
            test_dataset = RelationData(test_path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            #prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if global_threshold:
                prefix = args.test_data + '_global_threshold_'+ str(thresholds_dict['GLOBAL'])
                save_output_path = os.path.join("threshold_" + str(thresholds_dict['GLOBAL']),args.output_dir)
                if not os.path.exists(save_output_path):
                    os.makedirs(save_output_path)
                    print("make new save output dir: ", save_output_path)
            else:
                prefix = args.test_data + '_individual_threshold'
                save_output_path = args.output_dir
            all_logits = False
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)

            result = evaluate_multilabel(args, model, eval_dataset=test_dataset, label_map=i_to_label, prefix=prefix, threshold=thresholds_dict, multilabel_data=multilabel_data,global_threshold=global_threshold)

            if all_logits:
                    prefix += "_all_preds"
            predict_output_file = os.path.join(save_output_path, str(global_step) + prefix + "_predict_output.txt")
            sources = [s for s in open(test_path + ".src", "r").read().split('\n')]
            if not multilabel_data:
                targets = [i_to_label[int(s.strip("\n"))] for s in open(test_path + ".trg").readlines()]
            else:
                lines = [s.strip("\n").split(',') for s in open(test_path + ".trg").readlines()]
                targets = []
                for line in lines:
                    to_labels = [i_to_label[int(i)] for i in line]
                    targets.append(",".join(to_labels))

            with open(predict_output_file,"w") as f:
                width_s = max([len(s) for s in sources]) + 2
                width_t = max([len(t) for t in targets]) + 2
                headers = ["input_concepts", "gold_label", "predictions"]
                report = headers[0] + " "*(width_s - len(headers[0])) + \
                                           headers[1] + " "*(width_t - len(headers[1])) + headers[2]
                report += '\n\n'
                for s,t,p in zip(sources, targets, result["multi_predictions"]):
                    row = s + " "*(width_s - len(s)) + \
                                           t + " "*(width_t - len(t)) + str(p)
                    report += row
                    report +="\n"
                f.write(report)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())

            results.update(result)

    if args.do_multiple_prediction and args.local_rank in [-1, 0]:

        import csv
        thresholds_dict = dict()

        def to_float(str_num):
            new_str = '.'.join(str_num.split(','))
            return float(new_str)

        with open('thresholds.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                thresholds_dict.update({row[0].strip(): to_float(row[1])})

        global_threshold = True
        thresholds_dict['GLOBAL'] = 0.9
        # set other threshold to be 0.9
        #global_threshold = False
        random_tr = 0.9
        for key in thresholds_dict.keys():
            thresholds_dict[key] = 0.9
        thresholds_dict.update({'RANDOM':random_tr})

        test_path = os.path.join(args.data_dir, args.test_data)
        test_dataset = RelationData(test_path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer,targets_exit=False)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            #prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if global_threshold:
                prefix = args.test_data + '_global_threshold_' + str(thresholds_dict['GLOBAL'])
            else:
                #prefix = args.test_data + '_individual_threshold'
                prefix = args.test_data +'_global_threshold_' +str(thresholds_dict['GLOBAL']) + '_RANDOM_threshold_' + str(thresholds_dict['RANDOM'])
            all_logits = False
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)

            result = get_prediction_multilabel(args, model, eval_dataset=test_dataset, label_map=i_to_label, prefix=prefix, threshold=thresholds_dict, global_threshold=global_threshold)

            if all_logits:
                    prefix += "_all_preds"
            prefix_max = args.test_data + '_max'
            predict_output_file = os.path.join(args.output_dir, str(global_step) + prefix + "_predict_output.txt")
            predict_output_file_max = os.path.join(args.output_dir, str(global_step) + prefix_max + "_predict_output.txt")
            sources = [s for s in open(test_path + ".src", "r").read().split('\n')]

            with open(predict_output_file,"w") as f:
                width_s = max([len(s) for s in sources]) + 2
                #width_t = max([len(t) for t in targets]) + 2
                headers = ["input_concepts",  "predictions"]
                report = headers[0] + " "*(width_s - len(headers[0])) + headers[1]
                report += '\n\n'
                for s,p in zip(sources, result["multi_predictions"]):
                    row = s + " "*(width_s - len(s))  + str(p)
                    report += row
                    report +="\n"
                f.write(report)

            with open(predict_output_file_max,"w") as f:
                width_s = max([len(s) for s in sources]) + 2
                #width_t = max([len(t) for t in targets]) + 2
                headers = ["input_concepts",  "predictions"]
                report = headers[0] + " "*(width_s - len(headers[0])) + headers[1]
                report += '\n\n'
                for s,p in zip(sources, result["max_predictions"]):
                    row = s + " "*(width_s - len(s))  + str(p)
                    report += row
                    report +="\n"
                f.write(report)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())

            results.update(result)

    if args.do_max_prediction and args.local_rank in [-1, 0]:


        def to_float(str_num):
            new_str = '.'.join(str_num.split(','))
            return float(new_str)


        test_path = os.path.join(args.data_dir, args.test_data)
        test_dataset = RelationData(test_path, label_map=label_map, max_len=args.max_seq_length, tokenizer=tokenizer,targets_exit=False)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            #prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            prefix = args.test_data + '_max'


            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)

            result = get_prediction_max(args, model, eval_dataset=test_dataset, label_map=i_to_label, prefix=prefix)


            predict_output_file = os.path.join(args.output_dir, str(global_step) + prefix + "_predict_output.txt")
            sources = [s for s in open(test_path + ".src", "r").read().split('\n')]

            with open(predict_output_file,"w") as f:
                width_s = max([len(s) for s in sources]) + 2
                #width_t = max([len(t) for t in targets]) + 2
                headers = ["input_concepts",  "predictions"]
                report = headers[0] + " "*(width_s - len(headers[0])) + headers[1]
                report += '\n\n'
                for s,p in zip(sources, result["max_predictions"]):
                    row = s + " "*(width_s - len(s))  + str(p)
                    report += row
                    report +="\n"
                f.write(report)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())

            results.update(result)
    return results


if __name__ == "__main__":
    main()