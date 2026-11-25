# -*- coding: utf-8 -*-
"""中文长文本摘要 + 关键词提取器（仅使用 Python 标准库，无需训练）。

算法概要
--------
1. 分句：按 。！？；和换行切分，过滤去标点后长度 < 8 的句子。
2. 句向量：字符 2-gram 词袋 + 自算 TF-IDF，句子相似度 = 余弦。
3. 摘要：以句子为节点、相似度 > 0 为边建图，跑 TextRank（阻尼 0.85，
   最多迭代 50 次 / 收敛阈值 1e-4），取权重 Top-3 后按原文顺序输出。
4. 关键词：字符 2-gram 为节点、共现窗口 5 个字为边建图，跑带权 PageRank，
   过滤停用字/词后取 Top-5，并把原文中相邻的高分 bigram 合并成词。
5. 评测：关键句命中率（bigram 余弦 >= 0.6 视为命中）、字符级 ROUGE-1、
   压缩率（摘要字数 / 原文总字数，均按去标点字符计）。

用法
----
    python main.py --eval
    python main.py --demo
    python main.py --summarize data/sample2.txt
"""

import sys
import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict

# Windows 控制台默认可能是 GBK，这里强制按 UTF-8 输出，避免中文报错。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 自建停用字/词表（44 条，不依赖任何外部词表）
STOPWORDS = [
    "的", "了", "是", "在", "和", "与", "及", "等", "我", "你",
    "他", "她", "它", "我们", "这个", "那个", "这些", "那些", "一个", "一种",
    "可以", "能够", "进行", "通过", "以及", "但是", "而且", "因为", "所以", "如果",
    "虽然", "已经", "需要", "对于", "关于", "其中", "同时", "此外", "目前", "表示",
    "认为", "指出", "强调", "要求",
]
STOP_CHARS = {w for w in STOPWORDS if len(w) == 1}

# 需要从“字面”上剔除的标点/空白（中文常见标点 + 英文标点）
PUNCT_CHARS = (
    "，。！？；：、,.!?;:"
    "“”‘’「」『』（）()《》〈〉【】[]{}"
    "—…·~～|/\\+*#@&^%$=<>\"'`-_"
    " \t\r\n\u3000"
)
PUNCT_RE = re.compile("[" + re.escape(PUNCT_CHARS) + "]+")
SENT_RE = re.compile(r"[^。！？；\n\r]+[。！？；]?")

DAMPING = 0.85
MAX_ITER = 50
TOL = 1e-4
HIT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_chars(text):
    """去掉标点与空白，只保留文字/数字/字母，用于统计“字数”和 bigram。"""
    return PUNCT_RE.sub("", text)


def split_sentences(text):
    """按 。！？；和换行分句，保留句末标点；过滤去标点后长度 < 8 的句子。"""
    sentences = []
    for piece in SENT_RE.findall(text):
        piece = piece.strip()
        if not piece:
            continue
        if len(clean_chars(piece)) < 8:
            continue
        sentences.append(piece)
    return sentences


def char_bigrams(text):
    chars = clean_chars(text)
    return [chars[i:i + 2] for i in range(len(chars) - 1)]


def bigram_counter(text):
    return Counter(char_bigrams(text))


def cosine(v1, v2):
    if not v1 or not v2:
        return 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    dot = 0.0
    for key, value in v1.items():
        other = v2.get(key)
        if other:
            dot += value * other
    norm1 = math.sqrt(sum(v * v for v in v1.values()))
    norm2 = math.sqrt(sum(v * v for v in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def bigram_cosine(text_a, text_b):
    """原始字符 2-gram 词袋余弦（用于评测命中，不依赖摘要模型的 TF-IDF 权重）。"""
    return cosine(bigram_counter(text_a), bigram_counter(text_b))


# ---------------------------------------------------------------------------
# 句子向量与 TextRank
# ---------------------------------------------------------------------------
def build_tfidf_vectors(sentences):
    docs = [char_bigrams(s) for s in sentences]
    n = len(docs)
    df = Counter()
    for doc in docs:
        for gram in set(doc):
            df[gram] += 1
    idf = {gram: math.log((n + 1.0) / (count + 1.0)) + 1.0 for gram, count in df.items()}

    vectors = []
    for doc in docs:
        tf = Counter(doc)
        total = sum(tf.values()) or 1
        vec = {}
        for gram, count in tf.items():
            vec[gram] = (count / total) * idf[gram]
        vectors.append(vec)
    return vectors


def textrank_sentences(sentences):
    """在句子相似度图上跑 TextRank，返回 (句子权重, 相似度矩阵)。"""
    n = len(sentences)
    vectors = build_tfidf_vectors(sentences)

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            weight = cosine(vectors[i], vectors[j])
            sim[i][j] = weight
            sim[j][i] = weight

    out_sum = [sum(row) for row in sim]
    scores = [1.0] * n
    for _ in range(MAX_ITER):
        new_scores = [0.0] * n
        for i in range(n):
            acc = 0.0
            for j in range(n):
                if sim[j][i] > 0.0 and out_sum[j] > 0.0:
                    acc += (sim[j][i] / out_sum[j]) * scores[j]
            new_scores[i] = (1.0 - DAMPING) + DAMPING * acc
        delta = max(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if delta < TOL:
            break
    return scores, sim


# ---------------------------------------------------------------------------
# 关键词：字符 2-gram 共现图 + PageRank + 相邻合并
# ---------------------------------------------------------------------------
def is_valid_bigram(gram):
    if len(gram) != 2:
        return False
    for ch in gram:
        if not ("\u4e00" <= ch <= "\u9fff"):
            return False
    if gram in STOPWORDS:
        return False
    if any(ch in STOP_CHARS for ch in gram):
        return False
    return True


def weighted_pagerank(nodes, edges, damping=DAMPING, max_iter=MAX_ITER, tol=1e-6):
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    out_weight = defaultdict(float)
    incoming = defaultdict(list)
    for (src, dst), weight in edges.items():
        out_weight[src] += weight
        incoming[dst].append((src, weight))

    scores = {node: 1.0 / n for node in nodes}
    for _ in range(max_iter):
        new_scores = {}
        for node in nodes:
            acc = 0.0
            for src, weight in incoming[node]:
                if out_weight[src] > 0.0:
                    acc += (weight / out_weight[src]) * scores[src]
            new_scores[node] = (1.0 - damping) + damping * acc
        delta = max(abs(new_scores[node] - scores[node]) for node in nodes)
        scores = new_scores
        if delta < tol:
            break
    return scores


def keyword_graph_scores(sentences, window=5):
    """构建字符 2-gram 共现图（同一句中起点相距不超过 window 个字的 bigram 连边）。"""
    nodes = set()
    edges = Counter()
    for sentence in sentences:
        chars = clean_chars(sentence)
        tokens = []
        for i in range(len(chars) - 1):
            gram = chars[i:i + 2]
            if is_valid_bigram(gram):
                tokens.append((i, gram))
                nodes.add(gram)
        for a in range(len(tokens)):
            pos_a, gram_a = tokens[a]
            for b in range(a + 1, len(tokens)):
                pos_b, gram_b = tokens[b]
                if pos_b - pos_a > window:
                    break
                if gram_a == gram_b:
                    continue
                edges[(gram_a, gram_b)] += 1.0
                edges[(gram_b, gram_a)] += 1.0
    return weighted_pagerank(nodes, edges)


def extract_keywords(sentences, top_k=5, window=5, candidate_limit=20):
    """先给 bigram 打分，再把原文中相邻的高分 bigram 合并成更完整的词。"""
    scores = keyword_graph_scores(sentences, window=window)
    if not scores:
        return []

    ranked = sorted(scores, key=scores.get, reverse=True)
    top_grams = set(ranked[:candidate_limit])

    # 在原文中把连续出现的高分 bigram 合并成一个候选词（不依赖词表）
    term_best = {}
    for sentence in sentences:
        chars = clean_chars(sentence)
        n = len(chars)
        i = 0
        while i + 1 < n:
            if chars[i:i + 2] not in top_grams:
                i += 1
                continue
            j = i
            while j + 1 < n and chars[j:j + 2] in top_grams:
                j += 1
            term = chars[i:j + 1]
            if len(term) >= 2:
                avg = sum(scores[chars[k:k + 2]] for k in range(i, j)) / (j - i)
                if term not in term_best or avg > term_best[term]:
                    term_best[term] = avg
            i = j

    ordered = sorted(term_best.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    selected = []
    for term, score in ordered:
        if any(term in chosen or chosen in term for chosen, _ in selected):
            continue
        selected.append((term, score))
        if len(selected) >= top_k:
            break

    # 候选词不足 top_k 时，用未入选的高分 bigram 补齐
    if len(selected) < top_k:
        for gram in ranked:
            if len(selected) >= top_k:
                break
            if any(gram in chosen or chosen in gram for chosen, _ in selected):
                continue
            selected.append((gram, scores[gram]))

    total = sum(score for _, score in selected) or 1.0
    return [(term, score / total) for term, score in selected]


# ---------------------------------------------------------------------------
# 分析入口
# ---------------------------------------------------------------------------
def analyze(text, sentence_top_k=3, keyword_top_k=5):
    sentences = split_sentences(text)
    if not sentences:
        return {
            "sentences": [],
            "summary": [],
            "keywords": [],
            "sentence_scores": [],
            "summary_chars": 0,
            "total_chars": len(clean_chars(text)),
            "compression": 0.0,
        }

    scores, _ = textrank_sentences(sentences)
    k = min(sentence_top_k, len(sentences))
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k]
    chosen = sorted(ranked)  # 按原文顺序输出
    summary = [sentences[i] for i in chosen]

    keywords = extract_keywords(sentences, top_k=keyword_top_k)

    summary_chars = sum(len(clean_chars(s)) for s in summary)
    total_chars = len(clean_chars(text))
    compression = summary_chars / total_chars if total_chars else 0.0

    return {
        "sentences": sentences,
        "summary": summary,
        "keywords": keywords,
        "sentence_scores": scores,
        "summary_chars": summary_chars,
        "total_chars": total_chars,
        "compression": compression,
    }


# ---------------------------------------------------------------------------
# 评测指标
# ---------------------------------------------------------------------------
def hit_rate(summary_sentences, gold_sentences, threshold=HIT_THRESHOLD):
    details = []
    hits = 0
    for gold in gold_sentences:
        best = 0.0
        for cand in summary_sentences:
            best = max(best, bigram_cosine(cand, gold))
        hit = best >= threshold
        if hit:
            hits += 1
        details.append((gold, best, hit))
    rate = hits / len(gold_sentences) if gold_sentences else 0.0
    return hits, len(gold_sentences), rate, details


def rouge1(summary_text, reference_text):
    """字符级 ROUGE-1：统计 unigram 重合（去标点）。"""
    summary = Counter(clean_chars(summary_text))
    reference = Counter(clean_chars(reference_text))
    overlap = sum((summary & reference).values())
    summary_total = sum(summary.values())
    reference_total = sum(reference.values())
    precision = overlap / summary_total if summary_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def gold_in_original(gold_sentences, original_text):
    cleaned_original = clean_chars(original_text)
    return sum(1 for g in gold_sentences if clean_chars(g) in cleaned_original)


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------
def print_summary_block(result):
    print("摘要（%d 句，按原文顺序）：" % len(result["summary"]))
    for idx, sentence in enumerate(result["summary"], 1):
        print("  %d. %s" % (idx, sentence))


def print_keywords_block(result):
    items = ["%s(%.3f)" % (term, weight) for term, weight in result["keywords"]]
    print("关键词（Top-5，权重已归一化）：" + " | ".join(items))


def cmd_eval():
    gold_path = os.path.join(DATA_DIR, "gold.json")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    names = ["sample1.txt", "sample2.txt", "sample3.txt"]
    rows = []

    print("=" * 64)
    print("中文长文本摘要器 · 评测（3 篇自建示例，gold 为人工标注）")
    print("说明：摘要为抽取式 3 句；命中 = 与标注句字符 2-gram 余弦 >= %.2f" % HIT_THRESHOLD)
    print("=" * 64)

    for index, name in enumerate(names, 1):
        path = os.path.join(DATA_DIR, name)
        text = read_text(path)
        result = analyze(text)
        item_gold = gold[name]

        hits, total_gold, rate, details = hit_rate(result["summary"], item_gold["key_sentences"])
        precision, recall, f1 = rouge1("".join(result["summary"]),
                                       "".join(item_gold["key_sentences"]))
        found = gold_in_original(item_gold["key_sentences"], text)

        rows.append({
            "name": name,
            "hit_rate": rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "compression": result["compression"],
        })

        print()
        print("----- [%d/%d] %s -----" % (index, len(names), name))
        print("原文：%d 句 / %d 字" % (len(result["sentences"]), result["total_chars"]))
        print_summary_block(result)
        print_keywords_block(result)
        print("关键句命中率：%d/%d = %.3f" % (hits, total_gold, rate))
        for gold_sentence, similarity, hit in details:
            mark = "命中" if hit else "未命中"
            print("  [%s %.3f] %s" % (mark, similarity, gold_sentence))
        print("ROUGE-1（字符级）：P=%.3f  R=%.3f  F1=%.3f" % (precision, recall, f1))
        print("压缩率：%.3f（摘要 %d 字 / 原文 %d 字，均去标点计）"
              % (result["compression"], result["summary_chars"], result["total_chars"]))
        print("标注校验：%d/%d 条 gold 关键句可在原文中找到" % (found, total_gold))

    avg_hit = sum(r["hit_rate"] for r in rows) / len(rows)
    avg_p = sum(r["precision"] for r in rows) / len(rows)
    avg_r = sum(r["recall"] for r in rows) / len(rows)
    avg_f1 = sum(r["f1"] for r in rows) / len(rows)
    avg_comp = sum(r["compression"] for r in rows) / len(rows)

    print()
    print("=" * 64)
    print("平均结果（%d 篇）" % len(rows))
    print("关键句命中率：%.3f" % avg_hit)
    print("ROUGE-1：P=%.3f  R=%.3f  F1=%.3f" % (avg_p, avg_r, avg_f1))
    print("压缩率：%.3f" % avg_comp)
    f1_ok = "在区间内" if 0.45 <= avg_f1 <= 0.70 else "不在区间内"
    comp_ok = "在区间内" if 0.15 <= avg_comp <= 0.35 else "不在区间内"
    print("参考区间：ROUGE-1 F1 ∈ [0.45, 0.70]（%s）；压缩率 ∈ [0.15, 0.35]（%s）"
          % (f1_ok, comp_ok))
    print("=" * 64)


def cmd_demo():
    path = os.path.join(DATA_DIR, "sample1.txt")
    text = read_text(path)
    result = analyze(text)

    print("=" * 64)
    print("中文长文本摘要器 · 效果演示（sample1.txt：校园 AI 训练营招募通知）")
    print("=" * 64)
    print("【原文】（%d 句 / %d 字）" % (len(result["sentences"]), result["total_chars"]))
    print(text.strip())
    print()
    print("【摘要】")
    print_summary_block(result)
    print()
    print_keywords_block(result)
    print("【压缩率】%.3f（摘要 %d 字 / 原文 %d 字）"
          % (result["compression"], result["summary_chars"], result["total_chars"]))
    print("=" * 64)


def cmd_summarize(path):
    if not os.path.isfile(path):
        print("找不到文件：%s" % path)
        sys.exit(1)
    text = read_text(path)
    result = analyze(text)
    if not result["sentences"]:
        print("文件里没有可分句的中文内容（去标点后每句至少 8 字）。")
        sys.exit(1)

    print("=" * 64)
    print("文件：%s（%d 句 / %d 字）" % (path, len(result["sentences"]), result["total_chars"]))
    print("=" * 64)
    print_summary_block(result)
    print()
    print_keywords_block(result)
    print("【压缩率】%.3f（摘要 %d 字 / 原文 %d 字）"
          % (result["compression"], result["summary_chars"], result["total_chars"]))
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="中文长文本摘要 + 关键词提取器（纯 Python 标准库，免训练）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval", action="store_true",
                       help="在 3 篇示例上跑评测，打印每篇指标与平均值")
    group.add_argument("--demo", action="store_true",
                       help="打印 sample1 的原文、摘要与关键词")
    group.add_argument("--summarize", metavar="TXT", help="对任意文本文件生成摘要与关键词")
    args = parser.parse_args()

    if args.eval:
        cmd_eval()
    elif args.demo:
        cmd_demo()
    else:
        cmd_summarize(args.summarize)


if __name__ == "__main__":
    main()
