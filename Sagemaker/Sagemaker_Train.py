from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_from_disk
from torch.utils.data import Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 8]

from fastformer import FastformerForCausalLM, FastformerLMConfig

import random
import logging
import sys
import argparse
import os
import torch



def train_model(dataset, model, training_args, subset=None):
    train_data = dataset["train"]
    eval_data = dataset["validation"]
    if subset is not None:
        train_data = Subset(train_data, list(range(subset)))

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=dataset["test"])
    
    return trainer



def format_num_param(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'{(total_params / 1e6):2.1f}M'



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--subset", type=int, default=None)


    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    dataset = load_from_disk(args.training_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    
    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
        logging_strategy = "epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
        logging_dir=f"{args.output_data_dir}/logs"
    )


    ### normal fastformer model and training ###
    # huggingface trainer seems to oddly apply label smoothing during validation so we won't use it
    config = FastformerLMConfig(
        hidden_size = 1024, vocab_size = 32100, n_heads = 8,
        max_position_embeddings = block_size, groups = 1, kernel_size = 4,
        convolve = False, num_hidden_layers = 8, hidden_dropout_prob = .1,
        initializer_range = .02, label_smoothing = 0
    )
    fast_model = FastformerForCausalLM(config)
    fast_trainer = train_model(dataset, fast_model, training_args, subset=args.subset)
    fast_trainer.save_model(os.path.join(args.model_dir, "fast_model"))


    ### convolutional fastformer model and training ###
    config.convolve = True

    # decrease layers to make room for the convolutional layer
    config.num_hidden_layers = 6

    conv_model = FastformerForCausalLM(config)
    conv_trainer = train_model(dataset, conv_model, training_args, subset=args.subset)
    conv_trainer.save_model(os.path.join(args.model_dir, "conv_model"))


    ### gpt model and training ###
    config = GPT2Config(
        n_embd = 1024, vocab_size=len(tokenizer),
        n_positions = block_size, n_layer = 8, n_head = 8,
        resid_pdrop = .1, embd_pdrop = .1, attn_pdrop = .1,
        use_cache = False 
    )
    gpt_model = GPT2LMHeadModel(config)
    gpt_trainer = train_model(dataset, gpt_model, training_args, subset=args.subset)
    gpt_trainer.save_model(os.path.join(args.model_dir, "gpt_model"))
    
    ### results compilation ###
    loss1 = pd.DataFrame(fast_trainer.state.log_history[1::2]).set_index("epoch")["eval_loss"]
    loss2 = pd.DataFrame(conv_trainer.state.log_history[1::2]).set_index("epoch")["eval_loss"]
    loss3 = pd.DataFrame(gpt_trainer.state.log_history[1::2]).set_index("epoch")["eval_loss"]
    results = pd.DataFrame({f"Additive Attention ({format_num_param(fast_model)} parameters)": loss1,
                        f"Convolutional Additive Attention ({format_num_param(conv_model)} parameters)": loss2,
                        f"GPT2 ({format_num_param(gpt_model)} parameters)": loss3
                       }).iloc[:-1] # last row is a repeat
    ppl_results = np.exp(results)
    ppl_results.to_csv(os.path.join(args.output_dir, "ppl_results.csv"))
    
    plt.ylabel("Validation Perplexity")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "results.png"))