from typing import Any, List, Tuple

import torch

from espnet.nets.scorer_interface import BatchScorerInterface
from datamodule.transforms import TextTransform

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

class CausalLM:
    def __init__(self, model_name: str):
        """Initialize language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to('cuda')

    def score(self, sentence: str):
        """Return probability distribution for the next token in the sentence."""
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(
                inputs["input_ids"].to('cuda'),
                attention_mask=inputs["attention_mask"].to('cuda'),
                ).logits  # (batch, seq_len, vocab_size)
        return F.softmax(logits[0, -1], dim=-1)  # (vocab_size, )

class LanguageModelScorer(BatchScorerInterface):
    """Length bonus in beam search."""

    def __init__(self, n_vocab: int, model_name: str="meta-llama/Llama-3.2-3B"):
        """Initialize class.

        Args:
            vocab (dic): dictionary from token id to string

        Stores:
            text_transform: class that maps str[token] to List[id]
            model: model that scores a string, sequence of tokens

        """
        ## Imports the class that maps List[id] to str[token]
        ## Import the language model that implements score(self, sentence)
        self.n_vocab = n_vocab
        self.text_transform = TextTransform()  # VM id -> sentence
        self.lm_model = CausalLM(model_name)   # sentence -> probs
        self.tokens = []                       # VM id -> LM tokens id

        for sentence in self.text_transform.token_list:
            sentence = sentence.replace("<eos>", "").replace("\u2581", " ").replace("<space>", " ")  # TODO: check this
            if sentence == "": sentence = "."
            token = self.lm_model.tokenizer(sentence, return_tensors="pt")["input_ids"][0]  # (num_LM_toekns, )
            self.tokens.append(token[0].item())  # NOTE: only using first token
        self.tokens = torch.tensor(self.tokens).to('cuda')

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.  ## the full sentence
            state: Scorer state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and None

        """
        ## Returns a list (tensor) where element i is the score
        ##   of the current sentence y when adding the token i
        sentence = self.text_transform.post_process(y).replace("<eos>", "")
        if sentence == "":  # return uniform dist
            return torch.ones(self.n_vocab, device=x.device, dtype=x.dtype)/self.n_vocab, None

        probs = self.lm_model.score(sentence)  # (LM_vocab_size, )
        new_probs = probs[self.tokens]  # (n_vocab, )
        return new_probs, None

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        ## TODO (optional): improve the parallelization
        B = ys.shape[0]
        batch_scores = torch.zeros((B, self.n_vocab), device=xs.device, dtype=xs.dtype)
        for b in range(B):
            batch_scores[b] = self.score(ys[b], None, xs[b])[0]
        return batch_scores, None
