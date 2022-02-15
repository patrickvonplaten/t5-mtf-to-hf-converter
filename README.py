import sys
import os

path = sys.argv[1]
name = path.split("/")[-1]

items = name.split("-")
name_big = "-".join([items[0].upper()] + [items[1].capitalize()] + [x.upper() for x in items[2:]])

model_type = items[2].upper()
if len(items) < 4:
    variation_yes_no = "no"
    variations = "."
else:
    variation_yes_no = "the following"
    variations = ":\n"
    for item in items[3:]:
        abbreviation = "".join([s for s in item if not s.isdigit()])
        numbers = "".join([s for s in item if s.isdigit()])
        variations += f"- **{abbreviation}** is **{numbers}**\n"

with open(os.path.join(path, "pytorch_model.bin"), "r") as f:
    lines = f.readlines()
    size = int(lines[-1].strip().split()[-1])

num_params = str(round(size / 10 ** 6 / 4, 2))
size_in_mb = str(round(size / 10 ** 6, 2))
size_in_mb_fp16 = str(round(size / 10 ** 6 / 2, 2))

readme = f"""---
language:
- en
datasets:
- c4
tags:
- deep-narrow
inference: false

license: apache-2.0
---

# {name_big} (Deep-Narrow version)

{name_big} is a variation of [Google's original T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) following the [T5 model architecture](https://huggingface.co/docs/transformers/model_doc/t5).
It is a *pretrained-only* checkpoint and was released with the
paper **[Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers](https://arxiv.org/abs/2109.10686)**
by *Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar, Hyung Won Chung, Sharan Narang, Dani Yogatama, Ashish Vaswani, Donald Metzler*.

In a nutshell, the paper indicates that a **Deep-Narrow** model architecture is favorable for **downstream** performance compared to other model architectures
of similar parameter count.

To quote the paper:

> We generally recommend a DeepNarrow strategy where the model’s depth is preferentially increased
> before considering any other forms of uniform scaling across other dimensions. This is largely due to
> how much depth influences the Pareto-frontier as shown in earlier sections of the paper. Specifically, a
> tall small (deep and narrow) model is generally more efficient compared to the base model. Likewise,
> a tall base model might also generally more efficient compared to a large model. We generally find
> that, regardless of size, even if absolute performance might increase as we continue to stack layers,
> the relative gain of Pareto-efficiency diminishes as we increase the layers, converging at 32 to 36
> layers. Finally, we note that our notion of efficiency here relates to any one compute dimension, i.e.,
> params, FLOPs or throughput (speed). We report all three key efficiency metrics (number of params,
> FLOPS and speed) and leave this decision to the practitioner to decide which compute dimension to
> consider.

To be more precise, *model depth* is defined as the number of transformer blocks that are stacked sequentially.
A sequence of word embeddings is therefore processed sequentially by each transformer block.

## Details model architecture

This model checkpoint - **{name}** - is of model type **{model_type.lower().capitalize()}** with {variation_yes_no} variations{variations}
It has **{num_params}** million parameters and thus requires *ca.* **{size_in_mb} MB** of memory in full precision (*fp32*)
 or  **{size_in_mb_fp16} MB** of memory in half precision (*fp16* or *bf16*).

A summary of the *original* T5 model architectures can be seen here:

| Model | nl (el/dl) | ff | dm | kv | nh | #Params|
| ----| ---- | ---- | ---- | ---- | ---- | ----|
| Tiny | 4/4 | 1024 | 256 | 32 | 4 | 16M|
| Mini | 4/4 | 1536 | 384 | 32 | 8 | 31M|
| Small | 6/6 | 2048 | 512 | 32 | 8 | 60M|
| Base | 12/12 | 3072 | 768 | 64 | 12 | 220M|
| Large | 24/24 | 4096 | 1024 | 64 | 16 | 738M|
| Xl | 24/24 | 16384 | 1024 | 128 | 32 | 3B|
| XXl | 24/24 | 65536 | 1024 | 128 | 128 | 11B|

whereas the following abbreviations are used:

| Abbreviation | Definition |
| ----| ---- |
| nl | Number of transformer blocks (depth) |
| dm | Dimension of embedding vector (output vector of transformers block) |
| kv | Dimension of key/value projection matrix |
| nh | Number of attention heads |
| ff | Dimension of intermediate vector within transformer block (size of feed-forward projection matrix) | 
| el | Number of transformer blocks in the encoder (encoder depth) | 
| dl | Number of transformer blocks in the decoder (decoder depth) | 
| sh | Signifies that attention heads are shared | 
| skv | Signifies that key-values projection matrices are tied | 

If a model checkpoint has no specific, *el* or *dl* than both the number of encoder- and decoder layers correspond to *nl*.

## Pre-Training

The checkpoint was pretrained on the [Colossal, Cleaned version of Common Crawl (C4)](https://huggingface.co/datasets/c4) for 524288 steps using 
the span-based masked language modeling (MLM) objective.

## Fine-Tuning

**Note**: This model is a **pretrained** checkpoint and has to be fine-tuned for practical usage.
The checkpoint was pretrained in English and is therefore only useful for English NLP tasks.
You can follow on of the following examples on how to fine-tune the model:

*PyTorch*:

- [Summarization](https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization)
- [Question Answering](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_seq2seq_qa.py)
- [Text Classification](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification) - *Note*: You will have to slightly adapt the training example here to make it work with an encoder-decoder model.

*Tensorflow*:

- [Summarization](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/summarization)
- [Text Classification](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/text-classification) - *Note*: You will have to slightly adapt the training example here to make it work with an encoder-decoder model.

*JAX/Flax*:

- [Summarization](https://github.com/huggingface/transformers/tree/master/examples/flax/summarization)
- [Text Classification](https://github.com/huggingface/transformers/tree/master/examples/flax/text-classification) - *Note*: You will have to slightly adapt the training example here to make it work with an encoder-decoder model.

## Downstream Performance

TODO: Add table if available

## Computational Complexity

TODO: Add table if available

## More information

We strongly recommend the reader to go carefully through the original paper **[Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers](https://arxiv.org/abs/2109.10686)** to get a more nuanced understanding of this model checkpoint.
As explained in the following [issue](https://github.com/google-research/google-research/issues/986#issuecomment-1035051145), checkpoints including the *sh* or *skv* 
model architecture variations have *not* been ported to Transformers as they are probably of limited practical usage and are lacking a more detailed description. Those checkpoints are kept [here](https://huggingface.co/NewT5SharedHeadsSharedKeyValues) as they might be ported potentially in the future."""

with open(os.path.join(path, "README.md"), "w") as f:
    f.write(readme)
