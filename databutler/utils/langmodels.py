"""
A collection of wrapper utilities around language model APIs like the one offered by OpenAI.
"""
import os
import time
from typing import Optional, List, Union, Dict

import attrs
import openai
import transformers

from databutler.utils import paths
from databutler.utils.logging import logger


_CTR = 0


@attrs.define(eq=False, repr=False)
class _KeyManager:
    """
    An internal manager that cycles between OPENAI keys to enable a net-increase in rate=limits.
    """
    keys: List[str]
    _cur_key_idx: int = attrs.field(init=False, default=0)

    def set_next_key(self) -> None:
        """
        Cycles between available API keys. This method should be called before any request.
        """
        cur_key = self.keys[self._cur_key_idx]
        self._cur_key_idx = (self._cur_key_idx + 1) % len(self.keys)
        openai.api_key = cur_key


_OPENAI_KEY_MGR: Optional[_KeyManager] = None


def _setup_openai_key() -> None:
    """
    Loads the OpenAI api_key from the designated path, or from the environment variable OPENAI_KEY if it exists.

    :raises RuntimeError: if neither the path exists, nor the environment variable is supplied.
    """
    global _OPENAI_KEY_MGR

    if _OPENAI_KEY_MGR is not None:
        return

    path_key = os.path.join(paths.get_user_home_dir_path(), ".databutler", "openai_key.txt")
    if os.path.exists(path_key):
        #  Great just load it from the file.
        with open(path_key, "r") as f:
            key_text = f.read().strip()

        if "\n" in key_text:
            keys = key_text.split('\n')
        else:
            keys = [key_text]

        _OPENAI_KEY_MGR = _KeyManager(keys=keys)

    else:
        #  Check if the environment variable OPENAI_KEY is set.
        key_text = os.getenv("OPENAI_KEY")
        if key_text is not None:
            #  Interpret
            if key_text.startswith("["):
                #  Interpret as a list
                try:
                    keys = eval(key_text)
                except:
                    raise ValueError("Could not understand environment variable OPENAI_KEY")

            elif "," in key_text:
                keys = key_text.split(",")

            else:
                keys = key_text.split()

            #  First, create the cache file.
            os.makedirs(os.path.dirname(path_key), exist_ok=True)
            with open(path_key, "w") as f:
                f.write("\n".join(keys))

            _OPENAI_KEY_MGR = _KeyManager(keys=keys)

        else:
            raise RuntimeError(
                f"Neither the environment variable OPENAI_KEY is set, nor the file {path_key} exists."
            )


@attrs.define
class OpenAICompletion:
    text: str
    logprob: Optional[float]
    finish_reason: str


@attrs.define
class OpenAICompletionResponse:
    completions: List[OpenAICompletion]
    timestamp: str
    model: str
    id: str


_CODEX_TOKENIZER: Optional[transformers.GPT2Tokenizer] = None


def codex_tokenize(text: str) -> Dict[str, Union[List[int], List[str]]]:
    global _CODEX_TOKENIZER
    if _CODEX_TOKENIZER is None:
        _CODEX_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained("SaulLu/codex-like-tokenizer")

    return {
        "token_ids": _CODEX_TOKENIZER(text)['input_ids'],
        "token_strs": _CODEX_TOKENIZER.tokenize(text)
    }


_GPT3_TOKENIZER: Optional[transformers.GPT2TokenizerFast] = None


def gpt3_tokenize(text: str) -> Dict[str, Union[List[int], List[str]]]:
    global _GPT3_TOKENIZER
    if _GPT3_TOKENIZER is None:
        _GPT3_TOKENIZER = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    return {
        "token_ids": _GPT3_TOKENIZER(text)['input_ids'],
        "token_strs": _GPT3_TOKENIZER.tokenize(text)
    }


def tokenize(text: str, engine: str) -> Dict[str, Union[List[int], List[str]]]:
    if engine.startswith("code-"):
        return codex_tokenize(text)
    else:
        return gpt3_tokenize(text)


def openai_completion(engine: str,
                      prompt: str,
                      temperature: float = 0,
                      num_completions: int = 1,
                      max_tokens: int = 64,
                      stop: Optional[Union[str, List[str]]] = None,
                      return_logprobs: bool = False,
                      *,
                      retry_wait_duration: int = 60,
                      max_retries: int = 5,
                      **completion_kwargs,
                      ) -> OpenAICompletionResponse:
    """
    Wraps the OpenAI completion API, primarily for handling errors in a retry loop.

    :param engine: A string corresponding to the name of the engine to use.
                   Refer to https://beta.openai.com/docs/engines for a list of available engines.
    :param prompt: A string representing the prompt to complete.
    :param temperature: A float for the temperature.
                        Defaults to 0.0 where the model gives the most well-defined answer.
                        A value of 1.0 means the model is the most creative.
    :param num_completions: An integer representing the number of completions to return. Note that this makes sense
                            only when the temperature is non-zero, as duplicates will be returned otherwise.
    :param max_tokens: An integer representing the maximum number of tokens in a valid completion.
    :param stop: An optional string or list of strings corresponding to the stop token(s) for a completion.
    :param return_logprobs: A boolean for whether to include the log-probability of the overall completion(s).
    :param retry_wait_duration: An integer representing the time in seconds between retry attempts.
    :param max_retries: An integer representing the maximum number of retries.
    :param completion_kwargs: Other keyword arguments accepted by openai.Completion.create.
                              See https://beta.openai.com/docs/api-reference/completions/create for a full list.

    :return: An OpenAICompletionResponse object containing the completions along with metadata.
    :raises InvalidRequestError: If the request is invalid. This can happen if the wrong model is specified, or invalid
        values for max_tokens, temperature, stop etc. are provided.
    """
    global _CTR
    #  Ensure the API-key is set up.
    _setup_openai_key()
    #  Set up the key, using load-balancing if multiple keys are available.
    _OPENAI_KEY_MGR.set_next_key()

    num_keys_tried = 0
    num_retries = 0
    while num_retries <= max_retries:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                stop=stop,
                logprobs=0 if return_logprobs else None,
                **completion_kwargs,
            )

        except openai.error.InvalidRequestError:
            raise

        except openai.error.OpenAIError as e:
            if num_keys_tried < len(_OPENAI_KEY_MGR.keys):
                #  Try with another key before sleeping.
                num_keys_tried += 1

            else:
                logger.exception(e)
                logger.info(f"OpenAIError: Waiting for {retry_wait_duration} seconds after {_CTR} requests")
                time.sleep(retry_wait_duration)
                num_keys_tried = 0
                num_retries += 1
                _CTR = 0

            #  Use the next key for the next request.
            _OPENAI_KEY_MGR.set_next_key()

        else:
            _CTR += 1
            result = OpenAICompletionResponse(
                completions=[
                    OpenAICompletion(
                        text=c['text'],
                        logprob=None,
                        finish_reason=c['finish_reason']
                    ) for c in response['choices']
                ],
                timestamp=response['created'],
                model=response['model'],
                id=response['id'],
            )

            if return_logprobs:
                #  We need to compute log-probability of the completion(s).
                for c, orig_c in zip(result.completions, response['choices']):
                    if orig_c['finish_reason'] == "stop":
                        stop_set = {stop} if isinstance(stop, str) else set(stop)
                        logprob_sum = 0
                        logprobs_entry = orig_c["logprobs"]
                        for token, log_prob in zip(logprobs_entry["tokens"], logprobs_entry["token_logprobs"]):
                            if token in stop_set:
                                break

                            logprob_sum += log_prob

                        c.logprob = logprob_sum

                    else:
                        c.logprob = sum(orig_c["token_logprobs"])

            return result
