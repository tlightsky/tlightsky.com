+++
title = "Karpathy Transformer Tokeinzer"
date = 2024-04-09
description = ""

[taxonomies]
tags = ["AI"]
+++


# [Video](https://www.youtube.com/watch?v=zduSFxRajkE)


* chars to vector (oneshot)
* words to vector
* Unicode code to vector
    * too big (149813)
* UTF8 - most common used
    * MegaByte: Predicting Million-byte Sequences with Multiscale Transformer
    * Still too big
* Byte pair encoding
    * aaabdaaabac
    * ZabdZabac (Z=aa)
    * ZYdZYac (Y=ab, Z=aa)
    * XdXac (X=ZY, Y=ab, Z=aa)

* GPT-2 Tokenizer: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
* [Kapathy Basic Tokenizer](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py)

# Some other info
在深度学习中，常用的词元化（tokenization）方法包括以下几种：

* 空格分词（Whitespace Tokenization）：将文本按照空格进行分割，将每个分割后的词作为一个词元。例如，"Hello world!"会被分割为[“Hello”, “world!”]。
* 分词器（Tokenizer）：使用专门的分词器工具，如NLTK（Natural Language Toolkit）或spaCy，在文本中识别和分割单词。这些分词器可以处理更复杂的分割规则，例如处理标点符号、缩写词等。
* n-gram分词：将文本切分为连续的n个词的组合，这些组合称为n-gram。常见的是二元（bigram）和三元（trigram）分词。例如，"Hello world!"的bigram分词为[“Hello world”, “world!”]。
* 字符级别分词（Character-level Tokenization）：将文本按照字符进行分割，将每个字符作为一个词元。这种方法对于处理字符级别的任务（如拼写检查、机器翻译等）很有用。
* 子词（Subword）分词：将单词切分为更小的单元，如词根、前缀或后缀等。这种方法可以处理未登录词（out-of-vocabulary）问题，并且对于具有复杂形态的语言（如德语、芬兰语）也很有效。常见的子词分词算法有Byte-Pair Encoding（BPE）和SentencePiece。
这些方法的选择取决于特定的任务和语言，不同的词元化方法可能适用于不同的场景。在使用深度学习进行文本处理时，需要根据具体情况选择合适的词元化方法。