"""
A collection of wrapper utilities around language model APIs like the one offered by OpenAI.
"""
import os
import time
from typing import Optional, List, Union

import openai

from databutler.utils import paths

_OPENAI_KEY_SET = False


def _setup_openai_key() -> None:
    """
    Loads the OpenAI api_key from the designated path, or from the environment variable OPENAI_KEY if it exists.

    :raises RuntimeError: if neither the path exists, nor the environment variable is supplied.
    """
    global _OPENAI_KEY_SET

    if _OPENAI_KEY_SET:
        return

    path_key = os.path.join(paths.get_user_home_dir_path(), ".databutler", "openai_key.txt")
    if os.path.exists(path_key):
        #  Great just load it from the file.
        with open(path_key, "r") as f:
            openai.api_key = f.read().strip()

        _OPENAI_KEY_SET = True

    else:
        #  Check if the environment variable OPENAI_KEY is set.
        key = os.getenv("OPENAI_KEY")
        if key is not None:
            #  First, create the cache file.
            os.makedirs(os.path.dirname(path_key), exist_ok=True)
            with open(path_key, "w") as f:
                f.write(key)

            openai.api_key = key
            _OPENAI_KEY_SET = True

        else:
            raise RuntimeError(
                f"Neither the environment variable OPENAI_KEY is set, nor the file {path_key} exists."
            )


def openai_completion(engine: str,
                      prompt: str,
                      temperature: float = 0,
                      num_completions: int = 1,
                      max_tokens: int = 64,
                      stop: Optional[Union[str, List[str]]] = None,
                      return_logprobs: bool = False,
                      retry_wait_duration: int = 60,
                      *,
                      max_retries: int = 5,
                      **completion_kwargs,
                      ):
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
    :return:
    """

    #  Ensure the API-key is set up.
    _setup_openai_key()

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

        except openai.error.OpenAIError:
            time.sleep(retry_wait_duration)
            num_retries += 1

        else:
            #  Get rid of irrelevant fields, and rename to better reflect the meaning.
            result = {
                "completions": response['choices'],
                "model": response['model'],
                "id": response['id'],
                "timestamp": response['created'],
            }

            result['completions'] = [
                {
                    "text": c['text'],
                    "logprob": None,
                    "finish_reason": c['finish_reason'],
                } for c in result['completions']
            ]

            if return_logprobs:
                #  We need to compute log-probability of the completion(s).
                for c, orig_c in zip(result['completions'], response['choices']):
                    if orig_c['finish_reason'] == "stop":
                        stop_set = {stop} if isinstance(stop, str) else set(stop)
                        logprob_sum = 0
                        logprobs_entry = orig_c["logprobs"]
                        for token, log_prob in zip(logprobs_entry["tokens"], logprobs_entry["token_logprobs"]):
                            if token in stop_set:
                                break

                            logprob_sum += log_prob

                        c["logprob"] = logprob_sum

                    else:
                        c["logprob"] = sum(orig_c["token_logprobs"])

            return result
