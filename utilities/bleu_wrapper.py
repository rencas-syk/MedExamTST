from utilities._bleu_local.bleu import compute_bleu
from utilities._bleu_local.tokenizer_13a import Tokenizer13a


def bleu_wrapper(predictions, references, tokenizer = Tokenizer13a(), max_order=4, smooth=False):
    """
    Wrapper function for computing bleu locally (no huggingface access)
    Provided by GitHub user "justin13601", posted on May 12, 2024, last accessed on June 16, 2024:
    https://github.com/huggingface/evaluate/issues/315#issuecomment-2106121992

    """
    if isinstance(references[0], str):
            references = [[ref] for ref in references]
    references = [[tokenizer(r) for r in ref] for ref in references]
    predictions = [tokenizer(p) for p in predictions]

    score = compute_bleu(reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth)

    (bleu, precisions, bp, ratio, translation_length, reference_length) = score

    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "length_ratio": ratio,
        "translation_length": translation_length,
        "reference_length": reference_length,
    }
