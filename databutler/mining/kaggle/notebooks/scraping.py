from typing import Dict, List

import bs4
import markdownify


def convert_kaggle_html_to_ipynb(html_src: str):
    soup = bs4.BeautifulSoup(html_src, features="lxml")

    cells: List[Dict] = []

    #  Kaggle may decide to change this, and things may stop working.
    for cell_elem in soup.find_all("div", attrs={"class": "inner_cell"}):
        for code_input in cell_elem.find_all("div", attrs={"class": "input_area"}):
            cells.append(
                {
                    "cell_type": "code",
                    "source": code_input.text.strip(),
                    "metadata": {},
                    "outputs": [],
                    "execution_count": 0,
                }
            )

        for md_input in cell_elem.find_all("div", attrs={"class": "rendered_html"}):
            #  Remove the anchor links Kaggle adds
            for a in md_input.find_all("a", attrs={"class": "anchor-link"}):
                a.decompose()

            inner_html = "".join([str(x) for x in md_input.contents])
            cells.append(
                {
                    "cell_type": "markdown",
                    "source": markdownify.markdownify(inner_html),
                    "metadata": {},
                }
            )

    return {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "name": "python",
                "mimetype": "text/x-python",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": cells,
    }
